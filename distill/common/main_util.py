import torch

from .constant import def_logger
from .file_util import check_if_exists, make_parent_dirs

logger = def_logger.getChild(__name__)

def load_ckpt(ckpt_file_path, model=None, optimizer=None, lr_scheduler=None, strict=True):
    if check_if_exists(ckpt_file_path):
        ckpt = torch.load(ckpt_file_path, map_location='cpu')
    elif isinstance(ckpt_file_path, str) and \
            (ckpt_file_path.startswith('https://') or ckpt_file_path.startswith('http://')):
        ckpt = torch.hub.load_state_dict_from_url(ckpt_file_path, map_location='cpu', progress=True)
    else:
        message = 'ckpt file path is None' if ckpt_file_path is None \
            else 'ckpt file is not found at `{}`'.format(ckpt_file_path)
        logger.info(message)
        return None, None

    if model is not None:
        if 'model' in ckpt:
            logger.info('Loading model parameters')
            if strict is None:
                model.load_state_dict(ckpt['model'], strict=strict)
            else:
                model.load_state_dict(ckpt['model'], strict=strict)
        elif optimizer is None and lr_scheduler is None:
            logger.info('Loading model parameters only')
            model.load_state_dict(ckpt, strict=strict)
        else:
            logger.warning('No model parameters found')

    if optimizer is not None:
        if 'optimizer' in ckpt:
            logger.info('Loading optimizer parameters')
            optimizer.load_state_dict(ckpt['optimizer'])
        elif model is None and lr_scheduler is None:
            logger.info('Loading optimizer parameters only')
            optimizer.load_state_dict(ckpt)
        else:
            logger.warning('No optimizer parameters found')

    if lr_scheduler is not None:
        if 'lr_scheduler' in ckpt:
            logger.info('Loading scheduler parameters')
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        elif model is None and optimizer is None:
            logger.info('Loading scheduler parameters only')
            lr_scheduler.load_state_dict(ckpt)
        else:
            logger.warning('No scheduler parameters found')
    return ckpt.get('best_value', 0.0), ckpt.get('args', None)


def save_ckpt(model, optimizer, lr_scheduler, best_value, args, output_file_path):
    make_parent_dirs(output_file_path)
    pass