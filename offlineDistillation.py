import argparse
import os

from distill.common.main_util import load_ckpt

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR
import torch
import tensorlayerx as tlx

from distill.common import yaml_util
from distill.misc.log import set_basic_log_config, MetricLogger, SmoothedValue, plot_metrics
from distill.modules.registry import get_model
from distill.core.distillation import DistillationBox
from distill.losses.high_level import get_model_logits
from distill.common.constant import def_logger
from distill.datasets.registry import get_dataset
from distill.optim.registry import get_optimizer, get_scheduler
from distill.core.training import get_training_box
from distill.core.distillation import get_distillation_box
from distill.common.main_util import set_seed
import time
from distill.losses.registry import get_high_level_loss
import os.path as osp
from itertools import product
import numpy as np


logger = def_logger.getChild(__name__)
set_seed(42)

def compute_accuracy(logits, y, metrics):
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst


def evaluate(model, data, log_freq=10, title=None, header='Val: '):
    model.eval()
    metric_logger = MetricLogger(delimiter='    ')
    metrics = tlx.metrics.Accuracy()
    logits = get_model_logits(model, data)
    val_logits = tlx.gather(logits, data['test_idx'])
    val_y = tlx.gather(data['y'], data['test_idx'])
    val_acc = compute_accuracy(val_logits, val_y, metrics)

    metric_logger.meters['val_acc'].update(val_acc)

    return val_acc


def load_model(model_config):
    model = get_model(model_config)
    # model = get_model(model_config['key'], **model_config['kwargs'])
    # logger.info(model)

    # src_ckpt_file_path = model_config.get('src_ckpt', None)
    # load_ckpt(src_ckpt_file_path, model=model, strict=True)
    return model


def train(teacher_model, student_model, config, args):

    dataset_config = config['dataset']

    models_config = config['models']
    teacher_model_config = models_config.get('teacher_model', None)
    student_model_config = models_config.get('student_model', None)
    metric_config = config['log_metric']

    distill_type = config['type']
    teacher_model_ckpt_path = config['models']['teacher_model'].get('src_ckpt', './resource/ckpt/default-' + config['models']['teacher_model']['key'] + '.npz')
    if distill_type == 'OfflineDistillation':
        train_config = config['train_teacher']
        training_box = get_training_box(teacher_model, dataset_config, train_config)

        data = training_box.data
        data['val_idx'] = training_box.val_data
        data['test_idx'] = training_box.test_data
        data['train_idx'] = training_box.train_data
        if not os.path.isfile(teacher_model_ckpt_path):
            best_val_acc = 0.0

            log_freq = train_config['log_freq']
           
            model = training_box.model
            
            metric_logger = MetricLogger(delimiter="  ")
            for item in metric_config['item']:
                metric_logger.add_meter(item, SmoothedValue(window_size=1, fmt='{value}'))
            for epoch in range(args.start_epoch, training_box.num_epochs):
                header = 'Epoch: [{}]'.format(epoch)
                # training_box.pre_epoch_process(epoch=epoch)
                for data in metric_logger.log_every(data, log_freq, header):
                    train_loss = training_box.train(data, get_model_logits(teacher_model, data))
                    # train_loss = training_box.train(teacher_model(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes']), teacher_model(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes']))

                    # compute_accuracy(tlx.gather(teacher_model(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes']), data['test_idx']), tlx.gather(data['y'], data['test_idx']), tlx.metrics.Accuracy())
                    val_acc = evaluate(teacher_model, data, log_freq=log_freq, header='Validation:')
                    metrics_to_update = {}
                    for item in metric_config['item']:
                        if item == 'loss' and 'train_loss' in locals():  # 检查是否需要记录 loss，并且确保 loss 变量存在
                            metrics_to_update['loss'] = train_loss.item()
                        if item == 'val_acc' and 'val_acc' in locals():  # 检查是否需要记录 val_acc，并且确保 val_acc 变量存在
                            metrics_to_update['val_acc'] = val_acc.item()
                    metric_logger.update(**metrics_to_update)

                if val_acc > best_val_acc:
                    logger.info('Best accuracy: {:.4f} -> {:.4f}'.format(best_val_acc, val_acc))
                    logger.info('Updating ckpt at {}'.format(teacher_model_ckpt_path))
                    best_val_acc = val_acc
                    # student_model.save_weights("./"+student_model.name+".npz", format='npz_dict')
                    teacher_model.save_weights(teacher_model_ckpt_path, format='npz_dict')
                    teacher_model_ckpt_path = teacher_model_ckpt_path
            # teacher_model.load_weights(config['models']['teacher_model']['src_ckpt'], format='npz_dict')
    else :
        raise Exception("expect distillation type OfflineDistillation but get {}".format(distill_type))

    teacher_model.load_weights(teacher_model_ckpt_path, format='npz_dict')

    logits = get_model_logits(teacher_model, data)
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = compute_accuracy(test_logits, test_y, tlx.metrics.Accuracy())

    logger.info('Teacher Model Test acc: {:.4f}'.format(test_acc))

#  --------------------------------------- 

    logger.info('Student Start training')
    train_config = config['train']
    student_model_config = config['models']['student_model']
    dst_ckpt_file_path = student_model_config['dst_ckpt']
    training_box = get_training_box(student_model, dataset_config, train_config)
    best_val_acc = 0.0

    log_freq = train_config['log_freq']
    data = training_box.data
    t_idx = tlx.concat([training_box.train_data, training_box.test_data, training_box.val_data], axis=0)
    data['t_idx'] = t_idx
    data['val_idx'] = training_box.val_data
    data['test_idx'] = training_box.test_data
    data['train_idx'] = training_box.train_data

    model = training_box.model

    metric_logger = MetricLogger(delimiter="  ")
    for item in metric_config['item']:
        metric_logger.add_meter(item, SmoothedValue(window_size=1, fmt='{value}'))
    for epoch in range(args.start_epoch, training_box.num_epochs):
        header = 'Epoch: [{}]'.format(epoch)
        # training_box.pre_epoch_process(epoch=epoch)
        for data in metric_logger.log_every(data, log_freq, header):
            train_loss = training_box.train(data, get_model_logits(teacher_model, data), student_logits = get_model_logits(student_model, data))

        # compute_accuracy(tlx.gather(teacher_model(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes']), data['test_idx']), tlx.gather(data['y'], data['test_idx']), tlx.metrics.Accuracy())
            val_acc = evaluate(student_model, data, log_freq=log_freq, header='Validation:')
            metrics_to_update = {}
            for item in metric_config['item']:
                if item == 'loss' and 'train_loss' in locals():  # 检查是否需要记录 loss，并且确保 loss 变量存在
                    metrics_to_update['loss'] = train_loss.item()
                if item == 'val_acc' and 'val_acc' in locals():  # 检查是否需要记录 val_acc，并且确保 val_acc 变量存在
                    metrics_to_update['val_acc'] = val_acc.item()
            metric_logger.update(**metrics_to_update)
        metric_logger.record_metrics()

        if val_acc > best_val_acc:
            logger.info('Best accuracy: {:.4f} -> {:.4f}'.format(best_val_acc, val_acc))
            logger.info('Updating ckpt at {}'.format(dst_ckpt_file_path))
            best_val_acc = val_acc
            # student_model.save_weights("./"+student_model.name+".npz", format='npz_dict')
            student_model.save_weights(dst_ckpt_file_path, format='npz_dict')

        # training_box.post_epoch_process()
    plot_metrics(metric_logger, save_dir=metric_config['save_dir'], file_name=metric_config['file_name'], show=True)

    # if distributed:
    #     dist.barrier()

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # logger.info('Training time {}'.format(total_time_str))
    # training_box.clean_modules()

    model.load_weights(dst_ckpt_file_path, format='npz_dict')
    logits = get_model_logits(model, data)
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc2 = compute_accuracy(test_logits, test_y, tlx.metrics.Accuracy())

    logger.info('Test acc: {:.4f}'.format(test_acc2))

    return test_acc, test_acc2




def main(args, pram_dict=None):
    set_basic_log_config()
    logger.info(args)
    config = yaml_util.load_yaml_file(os.path.abspath(os.path.expanduser(args.config)))
    # logger.info(config)

    models_config = config['models']
    teacher_model_config = models_config.get('teacher_model', None)
    student_model_config = models_config.get('student_model', None)
    teacher_model = load_model(teacher_model_config) if teacher_model_config is not None else None
    student_model = load_model(student_model_config) if student_model_config is not None else None

    # teacher_model.load_weights(teacher_model_config['src_ckpt'], format='npz_dict')

    # src_ckpt_file_path = student_model_config.get('src_ckpt', None)
    # dst_ckpt_file_path = student_model_config['dst_ckpt']

    # dataset_config = config['dataset']

    if not args.test_only:
        test_acc, test_acc2 = train(teacher_model, student_model, config, args)
        return test_acc, test_acc2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge distillation for Graph Neural Networks')
    # parser.add_argument('--config', required=True, help='yaml file path') test_yaml glnn
    parser.add_argument('--config', default="/home/zgy/review/temp/distill/configs/glnn.yaml", help='yaml file path')
    parser.add_argument('--run_log', default="./test.log", help='log file path')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--epoch', default=100, type=int, metavar='N', help='num of epoch')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='num of epoch')
    parser.add_argument('--test_only', action='store_true', help='only test the models')

    args = parser.parse_args()

    param_grid = {
        # 'train.optimizer.kwargs.lr': [0.01, 0.001, 0.0001],
        'train.optimizer.kwargs.lr': [0.0001],
        'models.student_model.common_args.hidden_dim': [256],
        'models.student_model.common_args.drop_rate': [0.0]
    }
    iter = 2

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = product(*param_values)

    results = []
    for param_combination in all_combinations:
        test_acc_list = []
        test_acc2_list = []
        param_dict = dict(zip(param_names, param_combination))
        for i in range(iter):
            test_acc, test_acc2 = main(args, param_dict)
            test_acc_list.append(test_acc)
            test_acc2_list.append(test_acc2)

        results.append({
            "param_dict": param_dict,
            "test_acc_mean": np.mean(test_acc_list),
            "test_acc_std": np.std(test_acc_list),
            "test_acc2_mean": np.mean(test_acc2_list),
            "test_acc2_std": np.std(test_acc2_list)
        })
        # print(f"参数组合: {param_dict} 测试结果: {np.mean(test_acc_list)}±{np.std(test_acc_list)} --> {np.mean(test_acc2_list)}±{np.std(test_acc2_list)}")

    print("\n==== 所有参数组合的测试结果 ====")
    for result in results:
        param_dict = result['param_dict']
        print(f"参数组合: {param_dict} 测试结果: "
              f"{result['test_acc_mean']:.4f}±{result['test_acc_std']:.4f} "
              f"--> {result['test_acc2_mean']:.4f}±{result['test_acc2_std']:.4f}")

    # main(args)
