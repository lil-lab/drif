import os
import json

from torch.autograd import Variable
from tensorboardX import SummaryWriter

"""
A subclass of the summary writer that spies on all the scalar values that have been added and writes them to json files
in a sibling directory to the original logdir.
This is useful for later accessing the generating plots and curves in order to make plots for papers with pyplot
"""
class LoggingSummaryWriter(SummaryWriter):

    def __init__(self, log_dir="runs", restore=False, spy=False):
        super(LoggingSummaryWriter, self).__init__(log_dir)
        spl = os.path.split(log_dir)
        self.dirname = spl[0]
        self.runname = spl[1]
        self.spy = spy

        self.spy_dirname = os.path.join(self.dirname + "_spy", self.runname)
        os.makedirs(self.spy_dirname, exist_ok=True)

        self.value_file = os.path.join(self.spy_dirname, "values.json")
        self.index_file = os.path.join(self.spy_dirname, "indices.json")

        self.indices = {}
        self.values = {}

        if restore:
            if os.path.isfile(self.value_file) and os.path.isfile(self.index_file):
                with open(self.value_file, "r") as fp:
                    self.values = json.load(fp)
                with open(self.index_file, "r") as fp:
                    self.indices = json.load(fp)

        self.call = 0

    def add_scalar(self, tag, scalar_value, global_step=None):
        super(LoggingSummaryWriter, self).add_scalar(tag, scalar_value, global_step)
        self.spy_scalar(tag, scalar_value, global_step)

    def spy_scalar(self, tag, scalar_value, step):
        if not self.spy:
            return
        if tag not in self.values:
            self.values[tag] = []
            self.indices[tag] = []
        self.values[tag].append(scalar_value)
        self.indices[tag].append(step)

        self.call += 1
        if self.call < 10000:
            if self.call % 100 == 0:
                self.save_spied_values()
        else:
            if self.call % 1000 == 0:
                self.save_spied_values()

    def add_dict(self, prefix, dict, global_step=None):
        for key, value in dict.items():
            if type(value) == Variable:
                value = value.data
            if hasattr(value, "cuda"):
                value = value.cpu()[0]
            self.add_scalar(prefix + "/" + key, value, global_step)

    def save_spied_values(self):
        with open(self.index_file, "w") as fp:
            json.dump(self.indices, fp)
        with open(self.value_file, "w") as fp:
            json.dump(self.values, fp)