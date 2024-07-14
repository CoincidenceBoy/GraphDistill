import importlib
import sys
from ..common import misc_util
from distill.common.constant import def_logger

logger = def_logger.getChild(__name__)

MODEL_DICT = dict()
ADAPTATION_MODULE_DICT = dict()
AUXILIARY_MODEL_WRAPPER_DICT = dict()

# 动态导入 gammagl.modules 包
gammagl_datasets_module = importlib.import_module('gammagl.models')

# 确保 gammagl.datasets 模块现在在 sys.modules 中
assert 'gammagl.models' in sys.modules
MODEL_DICT.update(misc_util.get_classes_as_dict('gammagl.models'))

def get_model(key, *args, **kwargs):
    logger.info(MODEL_DICT)
    if key in MODEL_DICT:
        return MODEL_DICT[key](*args, **kwargs)
    raise ValueError('model_name `{}` is not expected'.format(key))

def register_model(arg=None, **kwargs):
    def _register_model(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        MODEL_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_model(arg)
    return _register_model