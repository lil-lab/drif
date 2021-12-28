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


class IncrementalMultimodalEmnlp(nn.Module):
    """
    pytorch module for final part of model
    combines embeddings of image, text, and previous action
    """
    def __init__(self, image_module, text_module, action_module,
                 max_episode_length, input_embedding_size, output_hidden_size, blocks_hidden_size, directions_hidden_size):

        #Intiialize nn.Module
        super(IncrementalMultimodalEmnlp, self).__init__()

        self.image_module = image_module
        self.text_module = text_module
        self.action_module = action_module

        # layer that converts from the concatenated inputs to h^1
        self.final_embedder = nn.Linear(
            input_embedding_size,
            output_hidden_size
        )

        # final output layers from h^1 to block and directions pre-sigmoid
        self.blocks_layer = nn.Linear(output_hidden_size, blocks_hidden_size)
        self.directions_layer = nn.Linear(output_hidden_size, directions_hidden_size)

    def init_weights(self):

        # Initializing weights
        self.apply(weights_init)
        # self.actor_linear.weight.data = self.normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = self.normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

    def forward(self, images, instruction, previous_action, model_state):

        # embed image
        images_embedded = self.image_module(images)

        # Text embedding doesn't change, so we keep this across everything
        if model_state is None:
            text_emb= self.text_module(instruction)
        else:
            text_emb= model_state

        # embed action
        action_emb = self.action_module(previous_action)

        new_model_state = text_emb

        # Concanate along embeddings
        pre_h1 = torch.cat((images_embedded, text_emb, action_emb), 1)

        # Create H^1
        h1 = F.relu(self.final_embedder(pre_h1))

        # Get outputs
        blocks_logit = self.blocks_layer(h1)
        direction_logit = self.directions_layer(h1)

        # Softmax over non-batch dimension
        block_logprob = F.log_softmax(blocks_logit, dim=1)  # 1 x num_block
        direction_logprob = F.log_softmax(direction_logit, dim=1)  # 1 x num_direction

        # From dipendra - properly handle STOP case and convert to 0-81 action space
        action_logprob = block_logprob.transpose(0, 1) + direction_logprob[:, :4]  # num_block x num_direction
        action_logprob = action_logprob.view(1, -1)
        stop_logprob = direction_logprob[:, 4:5]

        # 80 actions + stop action
        action_logprob = torch.cat([action_logprob, stop_logprob], dim=1)

        return action_logprob, new_model_state

