import inspect
import os
import sys


def get_classes_as_dict(package_name, is_lower=False):
    """
    Gets classes in a given package as dict.

    :param package_name: package name.
    :type package_name: str
    :param is_lower: if True, use lowercase module names.
    :type is_lower: bool
    :return: dict of classes defined in the given package.
    :rtype: dict
    """
    members = get_classes(package_name, require_names=True)
    class_dict = dict()
    for name, obj in members:
        class_dict[name.lower() if is_lower else name] = obj
    return class_dict

def get_classes(package_name, require_names=False):
    """
    Gets classes in a given package.

    :param package_name: package name.
    :type package_name: str
    :param require_names: whether to preserve member names.
    :type require_names: bool
    :return: collection of classes defined in the given package.
    :rtype: list[(str, class)] or list[class]
    """
    members = inspect.getmembers(sys.modules[package_name], inspect.isclass)
    if require_names:
        return members
    return [obj for _, obj in members]