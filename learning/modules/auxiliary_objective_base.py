import torch
from torch import nn as nn

from learning.modules.cuda_module import CudaModule


class AuxiliaryObjective(CudaModule):
    def __init__(self, name, *inputs):
        super(AuxiliaryObjective, self).__init__()
        self.required_inputs = inputs
        self.name = name

    def get_name(self):
        return self.name

    def get_required_inputs(self):
        return self.required_inputs

    def cuda(self, device=None):
        CudaModule.cuda(self, device)
        return self