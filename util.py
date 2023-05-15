from collections import defaultdict
import numpy as np


class RunningStats:

    def __init__(self, stats=None) -> None:
        self.data = defaultdict(lambda: [])
        self.length = 0
        if not stats:
            stats = {
                'mean': np.mean,
                'std': np.std,
                'min': np.min,
                'max': np.max,
            }
        self.stats = stats

    def add(self, **kwargs) -> None:
        for k, v in kwargs.items():
            self.data[k].append(float(v))
        self.length += 1

    def summary(self):
        res = dict()
        for k, v in self.data.items():
            if not v:
                continue
            val = dict()
            for stat_name, stat_fn in self.stats.items():
                val[stat_name] = stat_fn(v)
            res[k] = val
        return res

    def print_summary(self):
        stats_res = self.summary()
        for k, v in stats_res.items():
            print(f'{k} |', end=' ')
            for stat_name, value in v.items():
                print(f'{stat_name}: {value:0.3f} ', end='')
            print('|')

    def reset(self):
        self.data = defaultdict()
        self.length = 0

    def flush_summary(self):
        self.print_summary()
        self.reset()
