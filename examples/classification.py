import argparse
import os

import torch

from common import yaml_util
from modules.registry import get_model
# 确保导入 gcn 模块以进行注册
import modules.gcn


def main(args):
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))
    device = torch.device(args.device)
    teacher_model_config = config['models']['teacher_model']
    student_model_config = config['models']['student_model']
    teacher_model = load_model(teacher_model_config, device)
    student_model = load_model(student_model_config, device)

    src_ckpt_file_path = student_model_config.get('src_ckpt', None)
    dst_ckpt_file_path = student_model_config['dst_ckpt']

def load_model(model_config, device):
    model = get_model(model_config['key'], **model_config['kwargs'])

    src_ckpt_file_path = model_config.get('src_ckpt', None)
    # load_ckpt(src_ckpt_file_path, model=model, strict=True)
    return model.to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge distillation for Graph Neural Networks')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--run_log', help='log file path')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--epoch', default=0, type=int, metavar='N', help='num of epoch')

    args = parser.parse_args()
    main(args)
