import os
import pickle
import sys
from pathlib import Path


def make_parent_dirs(file_path):
    """
    Makes parent directories.

    :param file_path: file path
    :type file_path: str
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
