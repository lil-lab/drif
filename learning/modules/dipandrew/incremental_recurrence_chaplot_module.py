import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from learning.inputs.common import empty_float_tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class IncrementalRecurrenceChaplotModule(nn.Module):
    def __init__(self, input_emb_dim, output_emb_dim):
        super(IncrementalRecurrenceChaplotModule, self).__init__()
        self.input_emb_dim = input_emb_dim
        self.output_emb_dim = output_emb_dim
        self.lstm = nn.LSTMCell(input_emb_dim, output_emb_dim)
        self.is_cuda = False
        self.cuda_device = None

    def cuda(self, device=None):
        self.is_cuda = True
        self.cuda_device = device

    def init_weights(self):
        self.apply(weights_init)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, input_vector, hidden_vectors):
        """
        @param image_vector: batch of sequence of image embedding
        @param hidden_vectors: hidden vectors for each batch """

        if hidden_vectors is None:
            dims = (1, self.output_emb_dim)
            hidden_vectors = (Variable(empty_float_tensor(dims, self.is_cuda, self.cuda_device)),
                              Variable(empty_float_tensor(dims, self.is_cuda, self.cuda_device)))
        new_hidden_vector = self.lstm(input_vector, hidden_vectors)

        return new_hidden_vector
