import torch
import torch.nn as nn
from torch.autograd import Variable

from learning.inputs.sequence import mask_tensors
from learning.inputs.common import empty_float_tensor
from learning.modules.cuda_module import CudaModule
from learning.modules.spatial_softmax_2d import SpatialSoftmax2d


class CrossEntropy2d(CudaModule):

    def __init__(self, run_name="", ang_weight=0.33, fwd_weight=0.33, stop_weight=0.33):
        super(CrossEntropy2d, self).__init__()
        self.softmax = SpatialSoftmax2d()
        self.logsoftmax = SpatialSoftmax2d(log=True)

    def cuda(self, device=None):
        CudaModule.cuda(self, device)
        self.softmax.cuda(device)
        self.logsoftmax.cuda(device)

    def forward(self, pred, labels):

        #x = - self.softmax(labels) * self.logsoftmax(pred)
        x = - labels * self.logsoftmax(pred)
        # Sum over spatial dimensions:
        x = x.sum(2).sum(2)
        # Average over channels and batches
        loss = torch.mean(x)
        return loss