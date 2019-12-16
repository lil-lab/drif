import pprint

import torch

# get around get time
from utils.GS_timing import micros

"""
A profiler used to time execution of code.
Every time "tick" is called, it adds the amount of elapsed time to the "key" accumulator
This allows timing multiple things simultaneously and keeping track of how much time each takes
"""
class SimpleProfilerDummy():
    def __init__(self, torch_sync=False, print=True):
        pass

    def reset(self):
        pass

    def tick(self, key):
        pass

    def loop(self):
        pass

    def print_stats(self, every_n_times=1):
        pass

class SimpleProfilerReal():
    def __init__(self, torch_sync=False, print=True):
        """
        When debugging GPU code, torch_sync must be true, because GPU and CPU computation is asynchronous
        :param torch_sync: If true, will call cuda synchronize.
        :param print: If true, will print stats when print_stats is called. Pass False to disable output for release code
        """
        self.time = micros()
        self.loops = 0
        self.times = {}
        self.avg_times = {}
        self.sync = torch_sync
        self.print = print
        self.print_time = 0

    def reset(self):
        self.time = micros()
        self.times = {}

    def tick(self, key):
        if key not in self.times:
            self.times[key] = 0

        if self.sync:
            torch.cuda.synchronize()

        now = micros()
        self.times[key] += now - self.time
        self.time = now

    def loop(self):
        self.loops += 1
        for key, time in self.times.items():
            self.avg_times[key] = self.times[key] / self.loops

    def print_stats(self, every_n_times=1):
        self.print_time += 1
        if self.print and self.print_time % every_n_times == 0:
            total_time = 0
            if len(self.avg_times) > 0:
                print("Avg times per loop: ")
                pprint.pprint(self.avg_times)
                for k,v in self.avg_times.items():
                    if k != "out":
                        total_time += v
                print(f"Total avg loop time: {total_time}")
            else:
                print("Cumulative times: ")
                pprint.pprint(self.times)

SimpleProfiler = SimpleProfilerReal
