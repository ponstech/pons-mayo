import os
import pandas as pd
from itertools import product
from torch.utils.tensorboard import SummaryWriter
import datetime
from srcs.utils import get_logger
import time
import torch
from collections import defaultdict, deque
import torch.distributed as dist


class TensorboardWriter():
    def __init__(self, log_dir, enabled):
        self.logger = get_logger('tensorboard-writer')
        self.writer = SummaryWriter(log_dir) if enabled else None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

        self.step = 0

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.timer = datetime.datetime.now()

    def set_step(self, step):
        self.step = step
        if step == 0:
            self.timer = datetime.datetime.now()
        else:
            duration = datetime.datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            attr = getattr(self.writer, name)
            return attr

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

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
            value=self.value)

class MetricLogger(object):
    def __init__(self, logger, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = logger

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

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

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    self.logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    self.logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

class BatchMetrics:
    def __init__(self, *keys, postfix='', writer=None):
        self.writer = writer
        self.postfix = postfix        
        if postfix:
            keys = [k+postfix for k in keys]
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.postfix:
            key = key + self.postfix
        if key not in self._data.index:
            self._data.loc[key] = [0, 0, 0]
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        if self.postfix:
            key = key + self.postfix
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    
    

class EpochMetrics:
    def __init__(self, metric_names, phases=('train', 'valid'), monitoring='off'):
        self.logger = get_logger('epoch-metrics')
        # setup pandas DataFrame with hierarchical columns
        self._data = pd.DataFrame({}) # TODO: add epoch duration
        self.monitor_mode, self.monitor_metric = self._parse_monitoring_mode(monitoring)
        self.topk_idx = []

    def minimizing_metric(self, idx):
        if self.monitor_mode == 'off':
            return 0
        try:
            metric = self._data[self.monitor_metric].loc[idx]
        except KeyError:
            self.logger.warning("Warning: Metric '{}' is not found. "
                                "Model performance monitoring is disabled.".format(self.monitor_metric))
            self.monitor_mode = 'off'
            return 0
        if self.monitor_mode == 'min':
            return metric
        else:
            return - metric

    def _parse_monitoring_mode(self, monitor_mode):
        if monitor_mode == 'off':
            return 'off', None
        else:
            monitor_mode, monitor_metric = monitor_mode.split()
            monitor_metric = tuple(monitor_metric.split('/'))
            assert monitor_mode in ['min', 'max']
        return monitor_mode, monitor_metric

    def is_improved(self):
        if self.monitor_mode == 'off':
            return True

        last_epoch = self._data.index[-1]
        best_epoch = self.topk_idx[0]
        return last_epoch == best_epoch

    def keep_topk_checkpt(self, checkpt_dir, k=3):
        """
        Keep top-k checkpoints by deleting k+1'th best epoch index from dataframe for every epoch.
        """
        if len(self.topk_idx) > k and self.monitor_mode != 'off':
            last_epoch = self._data.index[-1]
            self.topk_idx = self.topk_idx[:(k+1)]
            if last_epoch not in self.topk_idx:
                to_delete = last_epoch
            else:
                to_delete = self.topk_idx[-1]

            # delete checkpoint having out-of topk metric
            filename = str(checkpt_dir / 'checkpoint-epoch{}.pth'.format(to_delete.split('-')[1]))
            try:
                os.remove(filename)
            except FileNotFoundError:
                # this happens when current model is loaded from checkpoint
                # or target file is already removed somehow
                pass

    def update(self, epoch, result):
        epoch_idx = f'epoch-{epoch}'
        # 
        if len(self._data.columns) == 0:
            for k, v in result.items():
                self._data[k] = [v]
            self._data.set_axis([epoch_idx], axis=0)
        else:
            self._data.loc[epoch_idx] = [v for k, v in result.items()]

        self.topk_idx.append(epoch_idx)
        self.topk_idx = sorted(self.topk_idx, key=self.minimizing_metric)

    def latest(self):
        return self._data[-1:]

    def to_csv(self, save_path=None):
        self._data.to_csv(save_path)

    def __str__(self):
        return str(self._data)
