import torch
from torch import nn as nn

from learning.modules.cuda_module import CudaModule


class Identity(CudaModule):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x