import importlib
import sys
from ..common import misc_util
from distill.common.constant import def_logger

logger = def_logger.getChild(__name__)

DATASET_DICT = dict()
COLLATE_FUNC_DICT = dict()
SAMPLE_LOADER_DICT = dict()
BATCH_SAMPLER_DICT = dict()
TRANSFORM_DICT = dict()
DATASET_WRAPPER_DICT = dict()


# 动态导入 gammagl.datasets 包
gammagl_datasets_module = importlib.import_module('gammagl.datasets')
pyg_datasets_module = importlib.import_module('torch_geometric.datasets')

# 确保 gammagl.datasets 模块现在在 sys.modules 中
assert 'gammagl.datasets' in sys.modules
DATASET_DICT.update(misc_util.get_classes_as_dict('torch_geometric.datasets'))
DATASET_DICT.update(misc_util.get_classes_as_dict('gammagl.datasets'))


def get_dataset(key, *args, **kwargs):
    if key in DATASET_DICT:
        dataset_class = DATASET_DICT[key]
        # 检查该数据集类是否需要 'name' 参数，若不需要则删除
        if 'name' in kwargs:
            try:
                # 检测是否支持 'name' 参数
                return dataset_class(*args, **kwargs)
            except TypeError:
                # 如果抛出 TypeError，说明不需要 'name' 参数，因此删除它
                kwargs.pop('name')

        return dataset_class(*args, **kwargs)
    else:
        raise ValueError(f"dataset_name '{key}' is not expected")


def register_dataset(arg=None, **kwargs):
    def _register_dataset(cls_or_func):
        key = kwargs.get('key')
        if key is None:
            key = cls_or_func.__name__

        DATASET_DICT[key] = cls_or_func
        return cls_or_func

    if callable(arg):
        return _register_dataset(arg)
    return _register_dataset
