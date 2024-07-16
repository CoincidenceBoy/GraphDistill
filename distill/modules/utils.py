
from ..common.constant import def_logger

logger = def_logger.getChild(__name__)

def redesign_model(org_model, model_config, model_label, model_type='original'):
    frozen_module_path_set = set(model_config.get('frozen_modules', list()))
    module_paths = model_config.get('sequential', list())
    if not isinstance(module_paths, list) or len(module_paths) == 0:
        logger.info('Using the {} model'.format(model_type))
        if len(frozen_module_path_set) > 0:
            logger.info('Frozen module(s): {}'.format(frozen_module_path_set))

        return org_model


