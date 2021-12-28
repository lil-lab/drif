import torch
import torch.nn as nn
import numpy as np
from learning.models.wasserstein_critic_base import WassersteinCriticBase


class VisitationDiscriminator(WassersteinCriticBase):

    def __init__(self):
        super(VisitationDiscriminator, self).__init__()

        ndf = 32
        self.clip_value = 0.01
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=4, out_channels=ndf, kernel_size=5, stride=4, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.InstanceNorm2d(ndf * 2),

            # state size. (ndf*2) x 4 x 4
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 2 x 2
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=2, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 1 x 1
        )
        # +2 for the two oob positions
        self.regressor = nn.Linear((ndf * 4), 1)

    def forward(self, vdistribution):
        # Add P(oob) input
        input_extra = torch.ones_like(vdistribution.inner_distribution)
        input_extra = input_extra * vdistribution.outer_prob_mass[:, :, np.newaxis, np.newaxis]
        full_input = torch.cat([vdistribution.inner_distribution, input_extra], dim=1)

        features = self.main(full_input)
        features = features.view(features.size(0), -1)
        output = self.regressor(features)
        return output.view(-1, 1).squeeze(1)

    def calc_domain_loss(self, pred_distributions, label_distributions):
        self.clip_weights()
        pred_score = self(pred_distributions)
        label_score = self(label_distributions)

        # cf Wasserstein GAN paper. The critic tries to maximize this difference.
        loss_wass = torch.mean(pred_score) - torch.mean(label_score)

        total_loss = loss_wass

        self.inc_iter()

        return total_loss
