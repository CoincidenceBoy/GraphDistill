import argparse
import os

from distill.common.main_util import load_ckpt

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR
import torch
import tensorlayerx as tlx

from distill.common import yaml_util
from distill.misc.log import set_basic_log_config, MetricLogger, SmoothedValue
from distill.modules.registry import get_model
from distill.core.distillation import DistillationBox
from distill.common.constant import def_logger
from distill.datasets.registry import get_dataset
from distill.optim.registry import get_optimizer, get_scheduler
from distill.core.training import get_training_box
from distill.core.distillation import get_distillation_box
import time
from distill.losses.registry import get_high_level_loss

logger = def_logger.getChild(__name__)


def compute_accuracy(logits, y, metrics):
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst


def evaluate(model, data, log_freq=10, title=None, header='Val: '):
    model.eval()
    metric_logger = MetricLogger(delimiter='    ')
    metrics = tlx.metrics.Accuracy()
    logits = model(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
    val_logits = tlx.gather(logits, data['val_idx'])
    val_y = tlx.gather(data['y'], data['val_idx'])
    val_acc = compute_accuracy(val_logits, val_y, metrics)

    metric_logger.meters['val_acc'].update(val_acc)

    return val_acc


def load_model(model_config):
    model = get_model(model_config['key'], **model_config['kwargs'])
    # logger.info(model)

    # src_ckpt_file_path = model_config.get('src_ckpt', None)
    # load_ckpt(src_ckpt_file_path, model=model, strict=True)
    return model


def train(teacher_model, student_model, dataset_dict, src_ckpt_file_path, dst_ckpt_file_path, config, args):
    logger.info('Start training')
    train_config = config['train']
    training_box = get_training_box(student_model, dataset_dict, train_config)
    best_val_acc = 0.0

    log_freq = train_config['log_freq']
    data = training_box.data
    t_idx = tlx.concat([training_box.train_data, training_box.test_data, training_box.val_data], axis=0)
    data['t_idx'] = t_idx
    data['val_idx'] = training_box.val_data
    data['test_idx'] = training_box.test_data
    data['train_idx'] = training_box.train_data

    model = training_box.model
    
    for epoch in range(args.start_epoch, training_box.num_epochs):
        # training_box.pre_epoch_process(epoch=epoch)
        train_loss = training_box.train(data, teacher_model(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes']))

        # compute_accuracy(tlx.gather(teacher_model(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes']), data['test_idx']), tlx.gather(data['y'], data['test_idx']), tlx.metrics.Accuracy())
        val_acc = evaluate(student_model, data, log_freq=log_freq, header='Validation:')

        logger.info('Epoch: {:0>3d}     train loss: {:.4f}   val acc: {:.4f}'.format(epoch, train_loss, val_acc))
        if val_acc > best_val_acc:
            logger.info('Best accuracy: {:.4f} -> {:.4f}'.format(best_val_acc, val_acc))
            logger.info('Updating ckpt at {}'.format(dst_ckpt_file_path))
            best_val_acc = val_acc
            # student_model.save_weights("./"+student_model.name+".npz", format='npz_dict')
            student_model.save_weights(dst_ckpt_file_path, format='npz_dict')

        # training_box.post_epoch_process()

    # if distributed:
    #     dist.barrier()

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # logger.info('Training time {}'.format(total_time_str))
    # training_box.clean_modules()

    model.load_weights(dst_ckpt_file_path, format='npz_dict')
    logits = model(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
    test_logits = tlx.gather(logits, data['test_idx'])
    test_y = tlx.gather(data['y'], data['test_idx'])
    test_acc = compute_accuracy(test_logits, test_y, tlx.metrics.Accuracy())

    logger.info('Test acc: {:.4f}'.format(test_acc))




def main(args):
    set_basic_log_config()
    # logger.info(args)
    config = yaml_util.load_yaml_file(os.path.abspath(os.path.expanduser(args.config)))
    # logger.info(config)

    models_config = config['models']
    teacher_model_config = models_config.get('teacher_model', None)
    student_model_config = models_config.get('student_model', None)
    teacher_model = load_model(teacher_model_config) if teacher_model_config is not None else None
    student_model = load_model(student_model_config) if student_model_config is not None else None

    teacher_model.load_weights(teacher_model_config['src_ckpt'], format='npz_dict')

    src_ckpt_file_path = student_model_config.get('src_ckpt', None)
    dst_ckpt_file_path = student_model_config['dst_ckpt']

    dataset_config = config['dataset']

    if not args.test_only:
        train(teacher_model, student_model, dataset_config, src_ckpt_file_path, dst_ckpt_file_path, config, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge distillation for Graph Neural Networks')
    # parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--config', default="/home/zgy/review/yds/distill/configs/test_yaml.yaml", help='yaml file path')
    parser.add_argument('--run_log', default="./test.log", help='log file path')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--epoch', default=100, type=int, metavar='N', help='num of epoch')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='num of epoch')
    parser.add_argument('--test_only', action='store_true', help='only test the models')

    args = parser.parse_args()
    main(args)
