import argparse
import os

from distill.common.main_util import load_ckpt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TL_BACKEND'] = 'torch'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0:Output all; 1:Filter out INFO; 2:Filter out INFO and WARNING; 3:Filter out INFO, WARNING, and ERROR
import torch
import tensorlayerx as tlx

from distill.common import yaml_util
from distill.misc.log import set_basic_log_config, MetricLogger, SmoothedValue
from distill.modules.registry import get_model
from distill.core.distillation import DistillationBox
from distill.losses.high_level import get_model_logits
from distill.common.constant import def_logger
from distill.datasets.registry import get_dataset
from distill.optim.registry import get_optimizer, get_scheduler
from distill.core.training import get_training_box
from distill.core.distillation import get_distillation_box
import time
from distill.losses.registry import get_high_level_loss
import os.path as osp


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

    distill_type = config['type']
    teacher_model_ckpt_path = config['models']['teacher_model'].get('src_ckpt', './resource/ckpt/default-teacher_model.npz')
    if distill_type == 'OnlineDistillation':
        train_model1_config = config['train_teacher']
        training_box1 = get_training_box(teacher_model, dataset_config, train_model1_config)
        train_model2_config = config['train']
        training_box2 = get_training_box(student_model, dataset_config, train_model2_config)

        data = training_box1.data
        data['val_idx'] = training_box1.val_data
        data['test_idx'] = training_box1.test_data
        data['train_idx'] = training_box1.train_data

        model1 = training_box1.model
        model2 = training_box2.model

        best_acc_model1 = 0.0
        best_acc_model2 = 0.0

        for epoch in range(args.start_epoch, max(training_box1.num_epochs, 200)):

            train_loss1 = training_box1.train(data, get_model_logits(teacher_model, data))
            train_loss2 = training_box2.train(data, get_model_logits(student_model, data))

            # ReinforceAgent 决定知识蒸馏方向
            if training_box1.criterion.__class__.__name__ == 'FreeKDLoss':
                output_model1 = get_model_logits(teacher_model, data)
                output_model2 = get_model_logits(student_model, data)

                from distill.modules.FreeKDAgent import FreeKDAgent
                agent = FreeKDAgent(input_dim=config['models']['teacher_model']['common_args']['num_class'] * 2, hidden_dim=32)
                state = torch.cat([output_model1.detach(), output_model2.detach()], dim=1)
                node_action_probs, structure_action_probs = agent(state)
                node_actions = torch.multinomial(node_action_probs, 1).squeeze()
                structure_actions = torch.multinomial(structure_action_probs, 1).squeeze()

                # 节点级别蒸馏损失
                distillation_loss_gcn = 0
                distillation_loss_gat = 0
                T = 2.0  # 温度
                import torch.nn.functional as F
                for i in range(len(node_actions)):
                    if node_actions[i] == 0:  # Model1 -> Model2
                        distillation_loss_gcn += F.kl_div(F.log_softmax(output_model2[i] / T, dim=0),
                                                          F.softmax(output_model1.detach()[i] / T, dim=0),
                                                          reduction='batchmean') * (T * T)
                    else:  # Model2 -> Model1
                        distillation_loss_gat += F.kl_div(F.log_softmax(output_model1[i] / T, dim=0),
                                                          F.softmax(output_model2.detach()[i] / T, dim=0),
                                                          reduction='batchmean') * (T * T)

                # 结构级别蒸馏损失
                structure_loss_gcn = 0
                structure_loss_gat = 0

                # 计算结构级别蒸馏损失
                for i, action in enumerate(structure_actions):
                    neighbors = data['edge_index'][1][data['edge_index'][0] == i]
                    if action == 1:  # 传播该节点的局部结构
                        if node_actions[i] == 0:  # Model1 -> Model2
                            structure_loss_gcn += F.kl_div(F.log_softmax(output_model2[neighbors] / T, dim=-1),
                                                           F.softmax(output_model1.detach()[neighbors] / T, dim=-1),
                                                           reduction='batchmean') * (T * T)
                        else:  # Model2 -> Model1
                            structure_loss_gat += F.kl_div(F.log_softmax(output_model1[neighbors] / T, dim=-1),
                                                           F.softmax(output_model2.detach()[neighbors] / T, dim=-1),
                                                           reduction='batchmean') * (T * T)

                train_loss1 += training_box1.criterion.mu * distillation_loss_gcn.detach().cpu().numpy() + training_box1.criterion.ro * structure_loss_gcn.detach().cpu().numpy()
                train_loss2 += training_box1.criterion.mu * distillation_loss_gat.detach().cpu().numpy() + training_box1.criterion.ro * structure_loss_gat.detach().cpu().numpy()

            val_acc_model1 = evaluate(model1, data)
            val_acc_model2 = evaluate(model2, data)

            if val_acc_model1 > best_acc_model1:
                best_acc_model1 = val_acc_model1

            if val_acc_model2 > best_acc_model2:
                best_acc_model2 = val_acc_model2

            if epoch % 10 == 0:
                logger.info(
                    'Epoch: {}   Model1:{}  train loss: {:.4f}   val acc: {:.4f}'.format(
                        epoch, model1.__class__.__name__, train_loss1, val_acc_model1) + '\n' +
                    'Epoch: {}   Model2:{}  train loss: {:.4f}   val acc: {:.4f}'.format(
                        epoch, model2.__class__.__name__, train_loss2, val_acc_model2))

        print(f"Final Best Model1 Test Accuracy: {best_acc_model1:.4f}, Final Best Model2 Test Accuracy: {best_acc_model2:.4f}")
    else :
        raise Exception("expect distillation type OfflineDistillation but get {}".format(distill_type))



def main(args):
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
        train(teacher_model, student_model, config, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge distillation for Graph Neural Networks')
    # parser.add_argument('--config', required=True, help='yaml file path') test_yaml glnn
    parser.add_argument('--config', default="/home/zgy/review/yds/distill/configs/freeKD.yaml", help='yaml file path')
    parser.add_argument('--run_log', default="./test.log", help='log file path')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--epoch', default=100, type=int, metavar='N', help='num of epoch')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='num of epoch')
    parser.add_argument('--test_only', action='store_true', help='only test the models')

    args = parser.parse_args()
    main(args)
