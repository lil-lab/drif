import torch
from torch import nn


class MetricSimilarityBase(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        pass