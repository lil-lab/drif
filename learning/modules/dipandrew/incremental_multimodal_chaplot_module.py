import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class IncrementalMultimodalChaplotModule(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, image_module, text_module, image_recurrence_module,
                 max_episode_length, final_image_height, final_image_width):
        super(IncrementalMultimodalChaplotModule, self).__init__()
        self.image_module = image_module
        self.image_recurrence_module = image_recurrence_module
        self.text_module = text_module
        self.dense_read = nn.Linear(512, 2)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(max_episode_length+1, self.time_emb_dim)

        # A3C-LSTM layers
        self.final_image_height = final_image_height
        self.final_image_width = final_image_width
        self.linear = nn.Linear(64 * self.final_image_height * self.final_image_width, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 4)

    @staticmethod
    def normalized_columns_initializer(weights, std=1.0):
        out = torch.randn(weights.size())
        out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
        return out

    def init_weights(self):

        # Initializing weights
        self.apply(weights_init)
        # self.actor_linear.weight.data = self.normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = self.normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

    def forward(self, image, instructions, tx, model_state):

        image_emb_seq = self.image_module(image)
        image_emb = image_emb_seq[:, 0, :, :, :]

        if model_state is None:
            text_emb, _ = self.text_module(instructions)
            image_hidden_states = None
        else:
            text_emb, image_hidden_states = model_state

        assert image_emb.size() == text_emb.size()
        x = image_emb * text_emb
        x = x.view(x.size(0), -1)

        # A3C-LSTM
        x = F.relu(self.linear(x))
        new_image_hidden_states = self.image_recurrence_module(x, image_hidden_states)
        hx, cx = new_image_hidden_states

        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        new_model_state = (text_emb, new_image_hidden_states)

        actor_linear = self.actor_linear(x)
        critic_linear = self.critic_linear(x)

        actor_linear[:, 3] = torch.sigmoid(actor_linear[:, 3])

        return actor_linear, new_model_state
        #return F.log_softmax(actor_linear, dim=1)
