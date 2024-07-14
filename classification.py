import argparse
import os

import torch

from distill.common import yaml_util
from distill.misc.log import set_basic_log_config
from distill.modules.registry import get_model
from distill.core.distillation import DistillationBox
from distill.common.constant import def_logger
from distill.datasets.registry import get_dataset

logger = def_logger.getChild(__name__)

def main(args):
    set_basic_log_config()
    logger.info(args)
    config = yaml_util.load_yaml_file(os.path.abspath(os.path.expanduser(args.config)))
    logger.info(config)
    device = torch.device(args.device)
    dataset_dict = config['dataset']
    dataset = get_dataset(dataset_dict['key'], **dataset_dict['init']['kwargs'])
    print(dataset)
    teacher_model_config = config['models']['teacher_model']
    student_model_config = config['models']['student_model']
    teacher_model = load_model(teacher_model_config, device)
    student_model = load_model(student_model_config, device)

    src_ckpt_file_path = student_model_config.get('src_ckpt', None)
    dst_ckpt_file_path = student_model_config['dst_ckpt']

    dataset_config = config['dataset']

    if not args.test_only:
        train(teacher_model, student_model, dataset_config, src_ckpt_file_path, dst_ckpt_file_path,
              device, config, args)

def load_model(model_config, device):
    model = get_model(model_config['key'], **model_config['kwargs'])
    print(model)

    # src_ckpt_file_path = model_config.get('src_ckpt', None)
    # load_ckpt(src_ckpt_file_path, model=model, strict=True)
    return model.to(device)

def train(teacher_model, student_model, dataset_dict, src_ckpt_file_path, dst_ckpt_file_path, device, config, args):
    logger.info('Start training')
    train_config = config['train']
    training_box = DistillationBox(teacher_model, student_model, dataset_dict, train_config, device)


def evaluate(model, data_loader, device):
    pass



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
