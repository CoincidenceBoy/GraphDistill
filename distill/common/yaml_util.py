import os

import yaml



def load_yaml_file(yaml_file_path, custom_mode=True):
    if custom_mode:
        yaml.add_constructor('!expanduser', yaml_expanduser, Loader=yaml.FullLoader)
        yaml.add_constructor('!abspath', yaml_abspath, Loader=yaml.FullLoader)
        yaml.add_constructor('!join', yaml_join, Loader=yaml.FullLoader)
    with open(yaml_file_path, 'r') as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def yaml_join(loader, node):
    seq = loader.construct_sequence(node, deep=True)
    return ''.join([str(i) for i in seq])

def yaml_expanduser(loader, node):
    path = loader.construct_python_str(node)
    return os.path.expanduser(path)

def yaml_abspath(loader, node):
    path = loader.construct_python_str(node)
    return os.path.abspath(path)