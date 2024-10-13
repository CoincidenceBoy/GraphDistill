import datetime
import logging
import time
from collections import defaultdict, deque
from logging import FileHandler, Formatter

import torch

from ..common.constant import def_logger, LOGGING_FORMAT
from ..common.file_util import make_parent_dirs
# from ..common.main_util import is_dist_avail_and_initialized
import tensorlayerx as tlx
import matplotlib.pyplot as plt
import os

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


class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        self.history = []

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
    
    def record(self):
        if self.deque:
            self.history.append(self.global_avg)

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )



# TODO: 这个类里面的log_every方法可以改成不一定需要dataloader的
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            assert isinstance(v, (float, int)), f'`{k}` ({v}) should be either float or int'
            self.meters[k].update(v)

    def record_metrics(self):
        for meter in self.meters.values():
            meter.record()

    def get_history(self, name):
        if name in self.meters:
            return self.meters[name].history
        return []

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, log_freq, header=None):
        if not isinstance(iterable, (list, tuple, tlx.dataflow.DataLoader)):
            iterable = [iterable]

        i = 0
        if not header:
            header = ''

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'

        # if torch.cuda.is_available():
        #     log_msg = self.delimiter.join([
        #         header,
        #         '[{0' + space_fmt + '}/{1}]',
        #         'eta: {eta}',
        #         '{meters}',
        #         'time: {time}',
        #         'data: {data}',
        #         'max mem: {memory:.0f}'
        #     ])
        # else:
        #     log_msg = self.delimiter.join([
        #         header,
        #         '[{0' + space_fmt + '}/{1}]',
        #         'eta: {eta}',
        #         '{meters}',
        #         'time: {time}',
        #         'data: {data}'
        #     ])
        log_msg = self.delimiter.join([
                header,
                '{meters}',
                'time: {time}'
            ])

        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % log_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(log_msg.format(
                        i, len(iterable),
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
                else:
                    logger.info(log_msg.format(
                        i, len(iterable),
                        meters=str(self),
                        time=str(iter_time)))

            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # logger.info('{} Total time: {}'.format(header, total_time_str))


def plot_metrics(metric_logger, save_dir=None, file_name=None, show=True):
    """
    绘制 MetricLogger 中存储的各个指标随 epoch 变化的趋势。

    :param metric_logger: 一个 MetricLogger 实例，包含要绘制的指标
    :type metric_logger: MetricLogger
    :param save_dir: 图表保存的目录路径。如果为 None，则不保存
    :type save_dir: str 或 None
    :param show: 是否显示图表。如果为 False，只保存不显示
    :type show: bool
    """

    plt.figure(figsize=(10, 6))

    # 绘制每个指标的历史记录
    for name, meter in metric_logger.meters.items():
        epochs = range(1, len(meter.history) + 1)
        plt.plot(epochs, meter.history, label=name)

    plt.title('Training Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path)
        logger.info(f'Metrics plot saved to {save_path}')

    if show:
        plt.show()

    plt.close()
