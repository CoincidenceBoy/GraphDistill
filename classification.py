import argparse
import os
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

logger = def_logger.getChild(__name__)


def train_one_epoch(training_box, device, epoch, log_freq):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    samples = training_box.dataset.samples
    targets = training_box.dataset.targets
    supp_dicts = training_box.dataset.supp_dicts

    start_time = time.time()

    # for i, (data, target) in enumerate(zip(samples, targets, supp_dicts)):
        
        # loss = training_box.forward_process



def train(teacher_model, student_model, dataset_dict, src_ckpt_file_path, dst_ckpt_file_path, config, args):
    logger.info('Start training')
    train_config = config['train']
    # training_box = get_training_box(student_model, dataset_dict, train_config, lr_factor) if teacher_model is None \
    #     else get_distillation_box(teacher_model, student_model, dataset_dict, train_config, lr_factor)
    training_box = get_training_box(student_model, dataset_dict, train_config)
    best_val_top1_accuracy = 0.0
    # optimizer, lr_scheduler = training_box.optimizer, training_box.lr_scheduler
    # # if file_util.check_if_exists(src_ckpt_file_path):
    # #     best_val_top1_accuracy, _ = load_ckpt(src_ckpt_file_path, optimizer=optimizer, lr_scheduler=lr_scheduler)

    # log_freq = train_config['log_freq']
    # # student_model_without_ddp = student_model.module if module_util.check_if_wrapped(student_model) else student_model
    # start_time = time.time()
    # for epoch in range(args.start_epoch, training_box.num_epochs):
    #     training_box.pre_epoch_process(epoch=epoch)
    #     train_one_epoch(training_box, device, epoch, log_freq)
    #     val_top1_accuracy = evaluate(student_model, training_box.val_data_loader, device, device_ids, distributed,
    #                                  log_freq=log_freq, header='Validation:')
    #     if val_top1_accuracy > best_val_top1_accuracy and is_main_process():
    #         logger.info('Best top-1 accuracy: {:.4f} -> {:.4f}'.format(best_val_top1_accuracy, val_top1_accuracy))
    #         logger.info('Updating ckpt at {}'.format(dst_ckpt_file_path))
    #         best_val_top1_accuracy = val_top1_accuracy
    #         save_ckpt(student_model_without_ddp, optimizer, lr_scheduler,
    #                   best_val_top1_accuracy, args, dst_ckpt_file_path)
    #     training_box.post_epoch_process()

    # if distributed:
    #     dist.barrier()

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # logger.info('Training time {}'.format(total_time_str))
    # training_box.clean_modules()




def main(args):
    set_basic_log_config()
    logger.info(args)
    config = yaml_util.load_yaml_file(os.path.abspath(os.path.expanduser(args.config)))
    logger.info(config)
    # device = torch.device(args.device)
    dataset_dict = config['dataset']
    dataset = get_dataset(dataset_dict['key'], **dataset_dict['init']['kwargs'])
    logger.info(dataset)

    teacher_model_config = config['models']['teacher_model']
    student_model_config = config['models']['student_model']
    teacher_model = load_model(teacher_model_config)
    student_model = load_model(student_model_config)

    optimizer_dict = config['train']['optimizer']
    optimizer = get_optimizer(teacher_model, optimizer_dict['key'], **optimizer_dict['kwargs'])
    logger.info(optimizer)
    scheduler_dict = config['train']['scheduler']
    scheduler = get_scheduler(optimizer, scheduler_dict['key'], **scheduler_dict['kwargs'])

    logger.info(optimizer)
    logger.info(scheduler)

    src_ckpt_file_path = student_model_config.get('src_ckpt', None)
    dst_ckpt_file_path = student_model_config['dst_ckpt']

    dataset_config = config['dataset']

    if not args.test_only:
        train(teacher_model, student_model, dataset_config, src_ckpt_file_path, dst_ckpt_file_path, config, args)

def load_model(model_config):
    model = get_model(model_config['key'], **model_config['kwargs'])
    logger.info(model)

    # src_ckpt_file_path = model_config.get('src_ckpt', None)
    # load_ckpt(src_ckpt_file_path, model=model, strict=True)
    return model

# def train(teacher_model, student_model, dataset_dict, src_ckpt_file_path, dst_ckpt_file_path, device, config, args):
#     logger.info('Start training')
#     train_config = config['train']
#     training_box = DistillationBox(teacher_model, student_model, dataset_dict, train_config, device)


# def evaluate(model, data_loader, device):
#     pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge distillation for Graph Neural Networks')
    # parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--config', default="/home/zgy/review/yds/distill/configs/test_yaml.yaml", help='yaml file path')
    parser.add_argument('--run_log', default="./test.log", help='log file path')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--epoch', default=0, type=int, metavar='N', help='num of epoch')
    parser.add_argument('-test_only', action='store_true', help='only test the models')

    args = parser.parse_args()
    main(args)
