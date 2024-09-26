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

def get_model(model_config):
    key = model_config['key']
    common_args = model_config.get('common_args', {})
    special_args = model_config.get('special_args', {})

    # 使用预处理函数将参数转换为模型所需的格式
    kwargs = preprocess_args(key, common_args, special_args)
    if not kwargs:
        kwargs = model_config.get('kwargs', {})

    if key in MODEL_DICT:
        return MODEL_DICT[key](**kwargs)
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

def preprocess_args(model_key, common_args, special_args):
    # 参数映射表：根据不同模型，映射通用参数到特定参数
    param_map = {
        'GCNModel': {
            'feature_dim': 'feature_dim',
            'hidden_dim': 'hidden_dim',
            'num_class': 'num_class',
            'num_layers': 'num_layers',
            'drop_rate': 'drop_rate'
        },
        'GraphSAGE_Full_Model': {
            'feature_dim': 'in_feats',
            'hidden_dim': 'n_hidden',
            'num_class': 'n_classes',
            'num_layers': 'n_layers',
            'drop_rate': 'dropout',
            'activation': 'activation'
        },
        'GATModel': {
            'feature_dim': 'feature_dim',
            'hidden_dim': 'hidden_dim',
            'num_class': 'num_class',
            'num_layers': 'num_layers',
            'drop_rate': 'drop_rate'
        },
        'GATV2Model': {
            'feature_dim': 'feature_dim',
            'hidden_dim': 'hidden_dim',
            'num_class': 'num_class',
            'num_layers': 'num_layers',
            'drop_rate': 'drop_rate'
        },
        'GCNIIModel': {
            'feature_dim': 'feature_dim',
            'hidden_dim': 'hidden_dim',
            'num_class': 'num_class',
            'num_layers': 'num_layers',
            'drop_rate': 'drop_rate'
        },
        'APPNPModel': {
            'feature_dim': 'feature_dim',
            # APPNPModel 没有 hidden_dim
            'num_class': 'num_class',
            # APPNPModel 没有 num_layers
            'drop_rate': 'drop_rate'
        },
        'MLP': {
            'feature_dim': 'in_channels',
            'hidden_dim': 'hidden_channels',
            'num_class': 'out_channels',
            'num_layers': 'num_layers',
            'drop_rate': 'dropout',
            'activation': 'act'
        }
    }


    if model_key not in param_map:
        # 合并 common_args 和 special_args
        kwargs = {**common_args, **special_args}
        return kwargs

    # 获取当前模型的参数映射
    mapped_args = {}
    for common_key, model_specific_key in param_map[model_key].items():
        if common_key in common_args:
            mapped_args[model_specific_key] = common_args[common_key]

    # 合并 special_args
    mapped_args.update(special_args)
    
    return mapped_args
