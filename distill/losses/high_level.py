import torch
from torch import nn
import torch.nn.functional as F

import tensorlayerx as tlx
from tensorlayerx.model import WithLoss

from .registry import register_high_level_loss, get_mid_level_loss
from ..common.constant import def_logger
from .registry import get_low_level_loss

logger = def_logger.getChild(__name__)


def get_model_logits(net, data):
    if "GCN" in net.__class__.__name__ :
        logits = net(data['x'], data['edge_index'], None, data['num_nodes'])
    elif "SAGE" in net.__class__.__name__ :
        logits = net(data['x'], data['edge_index'])
    elif "GAT" in net.__class__.__name__ :
        logits = net(data['x'], data['edge_index'], data['num_nodes'])
    elif "APPNP" in net.__class__.__name__ :
        logits = net(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
    elif "MLP" in net.__class__.__name__ :
        logits = net(data['x'])
    else:
        raise ValueError(f"Unsupported network class: {net.__class__.__name__}")
    return logits

class AbstractLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(AbstractLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward function is not implemented')

    def __str__(self):
        raise NotImplementedError('forward function is not implemented')
    

@register_high_level_loss(key='CrossEntropy')
class CrossEntropyLoss(WithLoss):
    def __init__(self, net, loss_fn = 'softmax_cross_entropy_with_logits'):
        loss_fn = get_low_level_loss(loss_fn)
        super(CrossEntropyLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        self.net = net

    def forward(self, data, y):
        logits = get_model_logits(self.backbone_network, data)
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss



@register_high_level_loss(key='KD_Loss')
class GLNNLoss(WithLoss):
    def __init__(self, net, loss_fn = 'softmax_cross_entropy_with_logits', lambad=0.1):
        loss_fn = get_low_level_loss(loss_fn)
        super(GLNNLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        self.backbone = net
        self.loss = loss_fn
        self.lambad = lambad

    def forward(self, data, teacher_logits):
        student_logits = get_model_logits(self.backbone_network, data)
        train_y = tlx.gather(data['y'], data['t_idx'])
        train_teacher_logits = tlx.gather(teacher_logits, data['t_idx'])
        train_student_logits = tlx.gather(student_logits, data['t_idx'])
        # loss = self._loss_fn(train_y, train_student_logits, train_teacher_logits, 0)
        
        loss_l = self._loss_fn(train_student_logits, train_y)
        teacher_logits = train_teacher_logits
        student_logits = train_student_logits
        teacher_probs = tlx.softmax(teacher_logits)
        student_probs = tlx.softmax(student_logits)
        # compute KL divergence
        kl_div = tlx.reduce_sum(teacher_probs * (tlx.log(teacher_probs+1e-10) - tlx.log(student_probs+1e-10)), axis=-1)
        loss_t = tlx.reduce_mean(kl_div)
        return self.lambad * loss_l + (1 - self.lambad) * loss_t
    
    def __str__(self):
        desc = 'Loss = '
        desc += ' + ' + self.backbone.name + "loss:" + self.loss.__name__
        return desc


@register_high_level_loss(key='FreeKD_Loss')
class FreeKDLoss(WithLoss):
    def __init__(self, net, loss_fn = 'softmax_cross_entropy_with_logits', mu=0.5, ro=0.5):
        loss_fn = get_low_level_loss(loss_fn)
        super(FreeKDLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        self.net = net
        self.mu = mu * 0.01
        self.ro = ro * 0.01

    def forward(self, data, y):
        logits = get_model_logits(self.backbone_network, data)
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss
    
    def __str__(self):
        desc = 'Loss = '
        desc += ' + ' + self.backbone.name + "loss:" + self.loss.__name__
        return desc

@register_high_level_loss(key='SD_Loss')
class GNNSDLoss(WithLoss):
    def __init__(self, net, loss_fn = 'softmax_cross_entropy_with_logits', alpha=0.1, beta=0.01, gamma=0.01):
        loss_fn = get_low_level_loss(loss_fn)
        super(GNNSDLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        self.net = net
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, data, y, *args, **kwargs):
        logits = get_model_logits(self.backbone_network, data)
        train_logits = tlx.gather(logits, data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)

        if 'model_io_dict' in kwargs:
            model_io_dict = kwargs['model_io_dict'] 
            # 2. LL：浅层和深层logits蒸馏损失
            # 自适应差异保持(ADR)正则化：计算所有相邻层之间的NDR差异
            if len(model_io_dict['ndr_values']) >= 2:  # 确保有多层NDR值
                for i in range(len(model_io_dict['ndr_values']) - 1):
                    ndr1 = model_io_dict['ndr_values'][i]  # 第i层的NDR
                    ndr2 = model_io_dict['ndr_values'][i + 1]  # 第i+1层的NDR
                    adr_loss = torch.mean(F.relu(ndr1 - ndr2))  # 如果第i层的NDR大于第i+1层,则增加惩罚
                    loss += self.alpha * adr_loss  # 将ADR正则化项加入到总损失中
                    print(f"ADR Loss between layer {i + 1} and layer {i + 2}: {adr_loss.item():.4f}")

            # 3. LN：邻域差异率（NDR）的正则化项,防止特征的过度平滑
            # 假设 LN 的逻辑是对某一层的特征进行差异化约束
            if len(model_io_dict['ndr_values']) >= 1:
                ln_loss = torch.mean(torch.stack(model_io_dict['ndr_values']))  # 假设 LN 对所有层的NDR平均
                loss += self.beta * ln_loss  # 加入LN正则化项
                print(f"LN Loss (NDR regularization): {ln_loss.item():.4f}")

            # 4. LG：图级别嵌入的蒸馏损失
            # 这里需要模型输出的图嵌入表示,并计算图级别的损失。
            if 'graph_embeds' in model_io_dict:
                g_embeds = model_io_dict['graph_embeds']  # 取图嵌入表示
                if len(g_embeds) >= 2:
                    lg_loss = torch.mean((g_embeds[-1] - g_embeds[0]) ** 2)  # 两层图嵌入之间的蒸馏
                    loss += self.gamma * lg_loss  # 加入LG正则化项
                    print(f"LG Loss (Graph Embedding distillation): {lg_loss.item():.4f}")

            # 清空NDR缓存,以便下一次前向传播重新计算
            model_io_dict['ndr_values'].clear()

        return loss
    
    def __str__(self):
        desc = 'Loss = '
        desc += ' + ' + self.backbone.name + "loss:" + self.loss.__name__
        return desc


@register_high_level_loss
class WeightedSumLoss(AbstractLoss):
    def __init__(self, model_term=None, sub_terms=None, **kwargs):
        super().__init__(sub_terms=sub_terms, **kwargs)
        if model_term is None:
            model_term = dict()
        self.model_loss_factor = model_term.get('weight', None)

    def forward(self, io_dict, model_loss_dict, targets):
        loss_dict = dict()
        student_io_dict = io_dict['student']
        teacher_io_dict = io_dict['teacher']
        for loss_name, (criterion, factor) in self.term_dict.items():
            loss_dict[loss_name] = factor * criterion(student_io_dict, teacher_io_dict, targets)

        sub_total_loss = sum(loss for loss in loss_dict.values()) if len(loss_dict) > 0 else 0
        if self.model_loss_factor is None or \
                (isinstance(self.model_loss_factor, (int, float)) and self.model_loss_factor == 0):
            return sub_total_loss

        if isinstance(self.model_loss_factor, dict):
            model_loss = sum([self.model_loss_factor[k] * v for k, v in model_loss_dict.items()])
            return sub_total_loss + model_loss
        return sub_total_loss + self.model_loss_factor * sum(model_loss_dict.values() if len(model_loss_dict) > 0 else [])

    def __str__(self):
        desc = 'Loss = '
        tuple_list = [(self.model_loss_factor, 'ModelLoss')] \
            if self.model_loss_factor is not None and self.model_loss_factor != 0 else list()
        tuple_list.extend([(factor, criterion) for criterion, factor in self.term_dict.values()])
        desc += ' + '.join(['{} * {}'.format(factor, criterion) for factor, criterion in tuple_list])
        return desc
