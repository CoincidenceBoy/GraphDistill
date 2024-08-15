from torch import nn
import tensorlayerx as tlx
from tensorlayerx.model import WithLoss

from .registry import register_high_level_loss, get_mid_level_loss
from ..common.constant import def_logger
from .registry import get_low_level_loss

logger = def_logger.getChild(__name__)


class AbstractLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(AbstractLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward function is not implemented')

    def __str__(self):
        raise NotImplementedError('forward function is not implemented')


@register_high_level_loss(key='glnn_loss_func')
class GLNNLoss(WithLoss):
    def __init__(self, net, loss_fn):
        loss_fn = get_low_level_loss(loss_fn)
        super(GLNNLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        self.backbone = net
        self.loss = loss_fn

    def forward(self, data, teacher_logits):
        student_logits = self.backbone_network(data['x'], data['edge_index'], data['edge_weight'], data['num_nodes'])
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
        return 0 * loss_l + (1 - 0) * loss_t
    
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
