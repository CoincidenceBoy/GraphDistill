import inspect
import os
import sys
from types import ModuleType
import importlib


def get_classes_as_dict(package_name, is_lower=False):
    members = get_classes(package_name, require_names=True)
    class_dict = dict()
    for name, obj in members:
        class_dict[name.lower() if is_lower else name] = obj
    return class_dict

def get_classes(package_name, require_names=False):
    members = inspect.getmembers(sys.modules[package_name], inspect.isclass)
    if require_names:
        return members
    return [obj for _, obj in members]
