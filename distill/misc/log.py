import datetime
import logging
import time
from collections import defaultdict, deque
from logging import FileHandler, Formatter

import torch
import torch.distributed as dist

from ..common.constant import def_logger, LOGGING_FORMAT
from ..common.file_util import make_parent_dirs
# from ..common.main_util import is_dist_avail_and_initialized

logger = def_logger.getChild(__name__)


def set_basic_log_config():
    """
    Sets a default basic configuration for logging.
    """
    logging.basicConfig(
        format=LOGGING_FORMAT,
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO
    )


def setup_log_file(log_file_path):
    """
    Sets a file handler with ``log_file_path`` to write a log file.

    :param log_file_path: log file path.
    :type log_file_path: str
    """
    make_parent_dirs(log_file_path)
    fh = FileHandler(filename=log_file_path, mode='w')
    fh.setFormatter(Formatter(LOGGING_FORMAT))
    def_logger.addHandler(fh)