import os
import pickle
import sys
from pathlib import Path

def check_if_exists(file_path):
    """
    Checks if a file/dir exists.

    :param file_path: file/dir path
    :type file_path: str
    :return: True if the given file exists
    :rtype: bool
    """
    return file_path is not None and os.path.exists(file_path)

def make_parent_dirs(file_path):
    """
    Makes parent directories.

    :param file_path: file path
    :type file_path: str
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
