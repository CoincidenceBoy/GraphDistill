import os

import yaml
import tensorlayerx as tlx

from .constant import def_logger
from .main_util import import_get, import_call, import_call_method

logger = def_logger.getChild(__name__)


def yaml_join(loader, node):
    """
    Joins a sequence of strings.

    :param loader: yaml loader.
    :type loader: yaml.loader.FullLoader
    :param node: node.
    :type node: yaml.nodes.Node
    :return: joined string.
    :rtype: str
    """
    seq = loader.construct_sequence(node, deep=True)
    return ''.join([str(i) for i in seq])


def yaml_pathjoin(loader, node):
    """
    Joins a sequence of strings as a (file) path.

    :param loader: yaml loader.
    :type loader: yaml.loader.FullLoader
    :param node: node.
    :type node: yaml.nodes.Node
    :return: joined (file) path.
    :rtype: str
    """
    seq = loader.construct_sequence(node, deep=True)
    return os.path.expanduser(os.path.join(*[str(i) for i in seq]))


def yaml_expanduser(loader, node):
    """
    Applies os.path.expanduser to a (file) path.

    :param loader: yaml loader.
    :type loader: yaml.loader.FullLoader
    :param node: node.
    :type node: yaml.nodes.Node
    :return: (file) path.
    :rtype: str
    """
    path = loader.construct_python_str(node)
    return os.path.expanduser(path)


def yaml_abspath(loader, node):
    """
    Applies os.path.abspath to a (file) path.

    :param loader: yaml loader.
    :type loader: yaml.loader.FullLoader
    :param node: node.
    :type node: yaml.nodes.Node
    :return: (file) path.
    :rtype: str
    """
    path = loader.construct_python_str(node)
    return os.path.abspath(path)


def yaml_import_get(loader, node):
    """
    Imports module and get its attribute.

    :param loader: yaml loader.
    :type loader: yaml.loader.FullLoader
    :param node: node.
    :type node: yaml.nodes.Node
    :return: module attribute.
    :rtype: Any
    """
    entry = loader.construct_mapping(node, deep=True)
    return import_get(**entry)


def yaml_import_call(loader, node):
    """
    Imports module and call the module/function e.g., instantiation.

    :param loader: yaml loader.
    :type loader: yaml.loader.FullLoader
    :param node: node.
    :type node: yaml.nodes.Node
    :return: result of callable module.
    :rtype: Any
    """
    entry = loader.construct_mapping(node, deep=True)
    return import_call(**entry)


def yaml_import_call_method(loader, node):
    """
    Imports module and call its method.

    :param loader: yaml loader.
    :type loader: yaml.loader.FullLoader
    :param node: node.
    :type node: yaml.nodes.Node
    :return: result of callable module.
    :rtype: Any
    """
    entry = loader.construct_mapping(node, deep=True)
    return import_call_method(**entry)


def yaml_getattr(loader, node):
    """
    Gets an attribute of the first argument.

    :param loader: yaml loader.
    :type loader: yaml.loader.FullLoader
    :param node: node.
    :type node: yaml.nodes.Node
    :return: module attribute.
    :rtype: Any
    """
    args = loader.construct_sequence(node, deep=True)
    return getattr(*args)


def yaml_setattr(loader, node):
    """
    Sets an attribute to the first argument.

    :param loader: yaml loader.
    :type loader: yaml.loader.FullLoader
    :param node: node.
    :type node: yaml.nodes.Node
    :return: module attribute.
    :rtype: Any
    """
    args = loader.construct_sequence(node, deep=True)
    setattr(*args)
    return args[0]


def yaml_access_by_index_or_key(loader, node):
    """
    Obtains a value from a specified data

    :param loader: yaml loader.
    :type loader: yaml.loader.FullLoader
    :param node: node.
    :type node: yaml.nodes.Node
    :return: accessed object.
    :rtype: Any
    """
    entry = loader.construct_mapping(node, deep=True)
    data = entry['data']
    index_or_key = entry['index_or_key']
    return data[index_or_key]


def update_dict(d, keys, value):
    """
    递归更新字典中嵌套的值。
    
    :param d: 要更新的字典。
    :type d: dict
    :param keys: 表示路径的列表，路径用点分隔（例如 'train.optimizer.kwargs.lr'）。
    :type keys: list
    :param value: 要设置的新值。
    :type value: any
    """
    key = keys[0]
    if len(keys) == 1:
        d[key] = value
    else:
        if key not in d:
            d[key] = {}
        update_dict(d[key], keys[1:], value)


def load_yaml_file(yaml_file_path, custom_mode=True, param_updates=None):
    """
    Loads a yaml file optionally with convenient constructors.

    :param yaml_file_path: yaml file path.
    :type yaml_file_path: str
    :param custom_mode: if True, uses convenient constructors.
    :type custom_mode: bool
    :return: loaded PyYAML object.
    :rtype: Any
    """
    if custom_mode:
        yaml.add_constructor('!join', yaml_join, Loader=yaml.FullLoader)
        yaml.add_constructor('!pathjoin', yaml_pathjoin, Loader=yaml.FullLoader)
        yaml.add_constructor('!expanduser', yaml_expanduser, Loader=yaml.FullLoader)
        yaml.add_constructor('!abspath', yaml_abspath, Loader=yaml.FullLoader)
        yaml.add_constructor('!import_get', yaml_import_get, Loader=yaml.FullLoader)
        yaml.add_constructor('!import_call', yaml_import_call, Loader=yaml.FullLoader)
        yaml.add_constructor('!import_call_method', yaml_import_call_method, Loader=yaml.FullLoader)
        yaml.add_constructor('!getattr', yaml_getattr, Loader=yaml.FullLoader)
        yaml.add_constructor('!setattr', yaml_getattr, Loader=yaml.FullLoader)
        yaml.add_constructor('!access_by_index_or_key', yaml_access_by_index_or_key, Loader=yaml.FullLoader)
    yaml_dict = {}
    with open(yaml_file_path, 'r') as fp:
        yaml_dict = yaml.load(fp, Loader=yaml.FullLoader)

        # 如果提供了超参数更新，则动态更新配置
        if param_updates:
            for param_path, value in param_updates.items():
                keys = param_path.split('.')
                update_dict(yaml_dict, keys, value)

        return update_dataset_yaml(yaml_dict, yaml_dict.get('dataset', {}).get('init', {}).get('kwargs', {}).get('name', 'cora'))
    

def update_dataset_yaml(yaml_dict, dataset_name = 'cora'):
    dataset_name = dataset_name.lower()
    dataset_info_map = {
        'cora': {
            'nodes': 2708,
            'edges': 10556,
            'feature_dim': 1433,
            'num_class': 7
        },
        'citeseer': {
            'nodes': 3327,
            'edges': 9104,
            'feature_dim': 3703,
            'num_class': 6
        },
        'pubmed': {
            'nodes': 19717,
            'edges': 88648,
            'feature_dim': 500,
            'num_class': 3
        }
    }
    if dataset_name not in dataset_info_map:
        raise ValueError(f"Unexpected Dataset '{dataset_name}', please check in '{dataset_info_map}")

    if 'teacher_model' in yaml_dict.get('models', {}):
        teacher_model = yaml_dict['models']['teacher_model']
        if 'common_args' in teacher_model:
            teacher_model['common_args']['feature_dim'] = dataset_info_map[dataset_name]['feature_dim']
            teacher_model['common_args']['num_class'] = dataset_info_map[dataset_name]['num_class']

    if 'student_model' in yaml_dict.get('models', {}):
        student_model = yaml_dict['models']['student_model']
        if 'common_args' in student_model:
            student_model['common_args']['feature_dim'] = dataset_info_map[dataset_name]['feature_dim']
            student_model['common_args']['num_class'] = dataset_info_map[dataset_name]['num_class']

    return yaml_dict

