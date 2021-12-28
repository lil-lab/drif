import torch.nn as nn


class ActionSimpleModule(nn.Module):
    def __init__(self, num_actions, action_emb_size):
        super(ActionSimpleModule, self).__init__()
        self.action_emb = nn.Embedding(num_actions + 1, action_emb_size)

    def forward(self, prev_action):
        return self.action_emb(prev_action)