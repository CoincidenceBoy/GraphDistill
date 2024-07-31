import copy

# import torch
# from torch.utils.data import DataLoader, random_split
# from torch.utils.data.distributed import DistributedSampler

from gammagl.loader import DataLoader

from ..common.constant import def_logger
from gammagl.utils import mask_to_index
from distill.datasets.registry import get_dataset
# from ..datasets.registry import get_collate_func, get_batch_sampler, get_dataset_wrapper
# from ..datasets.wrapper import default_idx2subpath, BaseDatasetWrapper, CacheableDataset

logger = def_logger.getChild(__name__)


def build_data_loader(dataset, data_loader_config):


    # collate_fn这个函数还是需要的，图分类中可以把不同维度的embedding进行对齐，现在先把他注释了
    # collate_fn = get_collate_func(data_loader_config.get('collate_fn', None))
    collate_fn = None
    data_loader_kwargs = data_loader_config['kwargs']
    return DataLoader(dataset, collate_fn=collate_fn, **data_loader_kwargs)

def build_data_loaders(dataset, data_loader_configs, accelerator=None):
    data_loader_dict = dict()

    # graph = dataset[0]
    graph = get_dataset(dataset['key'], **dataset['init']['kwargs'])[0]
    train_idx = mask_to_index(graph.train_mask)
    test_idx = mask_to_index(graph.test_mask)
    val_idx = mask_to_index(graph.val_mask)

    for data_loader_config in data_loader_configs:
        dataset_id = data_loader_config.get('split', None)
        if dataset_id == 'train':
            data_loader_dict['train'] = build_data_loader(train_idx, data_loader_config)
        elif dataset_id == 'test':
            data_loader_dict['test'] = build_data_loader(test_idx, data_loader_config)
        elif dataset_id == 'val':
            data_loader_dict['val'] = build_data_loader(val_idx, data_loader_config)
    return data_loader_dict
