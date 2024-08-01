from ..common.constant import def_logger

logger = def_logger.getChild(__name__)


def set_hooks(model, unwrapped_org_model, model_config, io_dict):
    pair_list = list()
    forward_hook_config = model_config.get('forward_hook', dict())
    if len(forward_hook_config) == 0:
        return pair_list
    
def clear_io_dict(model_io_dict):
    for module_io_dict in model_io_dict.values():
        for sub_dict in list(module_io_dict.values()):
            sub_dict.clear()
