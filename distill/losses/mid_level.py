import math

import torch
from torch import nn

from ..common.constant import def_logger
from .registry import register_mid_level_loss

logger = def_logger.getChild(__name__)

@register_mid_level_loss
class KDLoss(nn.KLDivLoss):

    def __init__(self, student_module_path, student_module_io, teacher_module_path, teacher_module_io,
                 temperature, alpha=None, beta=None, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.student_module_path = student_module_path
        self.student_module_io = student_module_io
        self.teacher_module_path = teacher_module_path
        self.teacher_module_io = teacher_module_io
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1 - alpha if beta is None else beta
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction, **kwargs)

    def forward(self, student_io_dict, teacher_io_dict, targets=None, *args, **kwargs):
        student_logits = student_io_dict[self.student_module_path][self.student_module_io]
        teacher_logits = teacher_io_dict[self.teacher_module_path][self.teacher_module_io]
        soft_loss = super().forward(torch.log_softmax(student_logits / self.temperature, dim=1),
                                    torch.softmax(teacher_logits / self.temperature, dim=1))
        if self.alpha is None or self.alpha == 0 or targets is None:
            return soft_loss

        hard_loss = self.cross_entropy_loss(student_logits, targets)
        return self.alpha * hard_loss + self.beta * (self.temperature ** 2) * soft_loss