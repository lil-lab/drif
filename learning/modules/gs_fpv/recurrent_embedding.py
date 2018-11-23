import torch
from torch.nn import LSTM

from learning.inputs.common import cuda_var

from learning.modules.map_transformer_base import MapTransformerBase
from learning.modules.cuda_module import CudaModule
from learning.inputs.common import cuda_var

class RecurrentEmbedding(CudaModule):

    def __init__(self, input_size, hidden_size):
        super(RecurrentEmbedding, self).__init__()

        self.lstm = LSTM(input_size, hidden_size, 1, True, False, 0, False)
        self.hidden_size = hidden_size

        self.last_h = None
        self.last_c = None

        self.hidden_size = hidden_size
        self.reset()
        self.dbg_t = None
        self.seq = 0

    def init_weights(self):
        pass

    def reset(self):
        self.last_h = cuda_var(torch.zeros(1, 1, self.hidden_size), self.is_cuda, self.cuda_device)
        self.last_c = cuda_var(torch.zeros(1, 1, self.hidden_size), self.is_cuda, self.cuda_device)

    def cuda(self, device=None):
        CudaModule.cuda(self, device)
        self.lstm.cuda(device)
        return self

    def forward(self, inputs):
        outputs = self.lstm(inputs, (self.last_h, self.last_c))
        self.last_h = outputs[1][0]
        self.last_c = outputs[1][1]
        return outputs[0]