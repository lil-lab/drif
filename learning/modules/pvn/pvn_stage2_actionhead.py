import torch.nn as nn


class PVN_Stage2_ActionHead(nn.Module):
    """
    Outputs a 4-D action, where
    """
    def __init__(self, h2=128, action_size=16):
        super(PVN_Stage2_ActionHead, self).__init__()
        self.linear = nn.Linear(h2, action_size)

    def init_weights(self):
        pass

    def forward(self, features):
        x = self.linear(features)
        return x