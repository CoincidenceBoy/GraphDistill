import copy

# import torch
# from torch.utils.data import DataLoader, random_split
# from torch.utils.data.distributed import DistributedSampler
from tensorlayerx.utils.data import DataLoader

from ..common.constant import def_logger
# from ..datasets.registry import get_collate_func, get_batch_sampler, get_dataset_wrapper
# from ..datasets.wrapper import default_idx2subpath, BaseDatasetWrapper, CacheableDataset

logger = def_logger.getChild(__name__)


def build_data_loader(dataset, data_loader_config, distributed=False, accelerator=None):


    # collate_fn这个函数还是需要的，图分类中可以把不同维度的embedding进行对齐，现在先把他注释了
    # collate_fn = get_collate_func(data_loader_config.get('collate_fn', None))
    collate_fn = None
    data_loader_kwargs = data_loader_config['kwargs']
    return DataLoader(dataset, collate_fn=collate_fn, **data_loader_kwargs)
