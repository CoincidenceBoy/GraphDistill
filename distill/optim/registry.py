import importlib
import sys
import tensorlayerx as tlx
from tensorlayerx import nn

from ..common import misc_util

from ..common.constant import def_logger
logger = def_logger.getChild(__name__)


OPTIM_DICT = dict()
SCHEDULER_DICT = dict()

tlx_optim_module = importlib.import_module('tensorlayerx.optimizers')
tlx_scheduler_module = importlib.import_module('tensorlayerx.optimizers.lr')

assert 'tensorlayerx.optimizers' in sys.modules
assert 'tensorlayerx.optimizers.lr' in sys.modules

OPTIM_DICT.update(misc_util.get_classes_as_dict('tensorlayerx.optimizers'))
SCHEDULER_DICT.update(misc_util.get_classes_as_dict('tensorlayerx.optimizers.lr'))
# OPTIM_DICT.update(misc_util.get_classes_as_dict('torch.optim'))
# SCHEDULER_DICT.update(misc_util.get_classes_as_dict('torch.optim.lr_scheduler'))


def register_optimizer(arg=None, **kwargs):
    def _register_optimizer(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        OPTIM_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_optimizer(arg)
    return _register_optimizer


def register_scheduler(arg=None, **kwargs):
    def _register_scheduler(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        SCHEDULER_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_scheduler(arg)
    return _register_scheduler


def get_optimizer(module, key, filters_params=False, **kwargs):
    is_module = isinstance(module, nn.Module)
    if key in OPTIM_DICT:
        optim_cls_or_func = OPTIM_DICT[key](**kwargs)
        if is_module and filters_params:
            params = module.parameters() if is_module else module
            updatable_params = [p for p in params if p.requires_grad]
            # 检查并更新优化器的默认参数
            # if hasattr(optim_cls_or_func, 'defaults'):
            #     for param, value in kwargs.items():
            #         optim_cls_or_func.defaults[param] = value
            #         logger.info(f"Updated default parameter {param} to {value}")
            return optim_cls_or_func(updatable_params, **kwargs)
        # return optim_cls_or_func(module, **kwargs)
        return optim_cls_or_func
    raise ValueError('No optimizer `{}` registered'.format(key))


def get_scheduler(optimizer, key, *args, **kwargs):
    if key in SCHEDULER_DICT:
        return SCHEDULER_DICT[key](optimizer.lr, *args, **kwargs)
    raise ValueError('No scheduler `{}` registered'.format(key))
