import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from learning.inputs.common import empty_float_tensor, cuda_var
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextSimpleModule(nn.Module):
    """
    pytorch module for text part of model
    assumes input is two parts:
    (1) text_tokens is pytorch variable consisting of instructions in
        batch (dimensionality is BatchSize x SequenceLength
      - each element of sequence is integer giving word ID
      - SequenceLength is based on longest element of sequence
      - each non-longest sequence should be padded by 0's at the end,
        to make the tensor rectangular
      - sequences in batch should be sorted in descending-length order
    (2) text_length is pytorch tensor giving actual lengths of each sequence
        in batch (dimensionality is BatchSize)
    """
    def __init__(self, emb_dim, hidden_dim, vocab_size,
                 num_layers=1):
        super(TextSimpleModule, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers)

        self.is_cuda = False
        self.cuda_device = None

    def cuda(self, device=None):
        self.is_cuda = True
        self.cuda_device = device

    def forward(self, instructions_batch):
        token_lists, _ = instructions_batch
        batch_size = len(token_lists)
        dims = (self.num_layers, batch_size, self.hidden_dim)
        hidden = (Variable(empty_float_tensor(*dims, self.is_cuda, self.cuda_device)),
                  Variable(empty_float_tensor(*dims, self.is_cuda, self.cuda_device)))

        # pad text tokens with 0's
        text_lengths = np.array([len(tokens) for tokens in token_lists])
        tokens_batch = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            num_zeros = text_lengths[0] - text_lengths[i]
            tokens_batch[i] = token_lists[i] + [0] * num_zeros
        tokens_batch = cuda_var(torch.from_numpy(np.array(tokens_batch)))

        # swap so batch dimension is second, sequence dimension is first
        tokens_batch = tokens_batch.transpose(0, 1)
        emb_sentence = self.embedding(tokens_batch)
        packed_input = pack_padded_sequence(emb_sentence, text_lengths)
        lstm_out_packed, _ = self.lstm(packed_input, hidden)
        # return average output embedding
        lstm_out, seq_lengths = pad_packed_sequence(lstm_out_packed)
        lstm_out = lstm_out.transpose(0, 1)
        sum_emb_list = []
        for i, seq_out in enumerate(lstm_out):
            seq_len = seq_lengths[i]
            sum_emb = torch.sum(seq_out[:seq_len], 0) / seq_len
            sum_emb_list.append(sum_emb.view(1, -1))
        return torch.cat(sum_emb_list)

