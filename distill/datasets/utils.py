import copy

# import torch
# from torch.utils.data import DataLoader, random_split
# from torch.utils.data.distributed import DistributedSampler
from tensorlayerx.utils.data import DataLoader

from ..common.constant import def_logger
# from ..datasets.registry import get_collate_func, get_batch_sampler, get_dataset_wrapper
# from ..datasets.wrapper import default_idx2subpath, BaseDatasetWrapper, CacheableDataset

logger = def_logger.getChild(__name__)


def build_data_loader(dataset, data_loader_config):


    # collate_fn这个函数还是需要的，图分类中可以把不同维度的embedding进行对齐，现在先把他注释了
    # collate_fn = get_collate_func(data_loader_config.get('collate_fn', None))
    collate_fn = None
    data_loader_kwargs = data_loader_config['kwargs']
    return DataLoader(dataset, collate_fn=collate_fn, **data_loader_kwargs)

def build_data_loaders(dataset_dict, data_loader_configs, accelerator=None):
    data_loader_list = list()
    for data_loader_config in data_loader_configs:
        dataset_id = data_loader_config.get('dataset_id', None)
        data_loader = None if dataset_id is None or dataset_id not in dataset_dict \
            else build_data_loader(dataset_dict[dataset_id], data_loader_config)
        data_loader_list.append(data_loader)
    return data_loader_list
