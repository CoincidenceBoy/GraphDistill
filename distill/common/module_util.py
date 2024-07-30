from collections import OrderedDict

from tensorlayerx.nn import Sequential, ModuleList, Module, Parameter
from .constant import def_logger

logger = def_logger.getChild(__name__)

def check_if_wrapped(model):
    return False


def get_module(root_module, module_path):
    module_names = module_path.split('.')
    module = root_module
    for module_name in module_names:
        if not hasattr(module, module_name):
            if isinstance(module, (Sequential, ModuleList)) and module_name.lstrip('-').isnumeric():
                module = module[int(module_name)]
            else:
                logger.warning('`{}` of `{}` could not be reached in `{}`'.format(
                    module_name, module_path, type(root_module).__name__)
                )
                return None
        else:
            module = getattr(module, module_name)
    return module