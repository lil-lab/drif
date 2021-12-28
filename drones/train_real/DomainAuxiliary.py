import torch
from torch import nn as nn
from torch.autograd import Function
import numpy as np

from learning.modules.auxiliary_objective_base import AuxiliaryObjective


class AdversaryAuxiliary(AuxiliaryObjective):
    def __init__(self, name, adversarial_network, *inputs):
        super(AdversaryAuxiliary, self).__init__(name, *inputs)
        self.discriminator = adversarial_network
        self.wasserstein = self.discriminator.wasserstein

    def forward(self, features_real, features_sim):
        # Important: opposite loss of the discriminator
        if not self.discriminator.wasserstein:
            batch_size = len(features_real)

            features = torch.cat([features_real, features_sim])
            real_label = 1.
            sim_label = 0.

            labels_real = torch.full((batch_size,), real_label)
            labels_sim = torch.full((batch_size,), sim_label)
            labels = torch.cat([labels_real, labels_sim])

            predictions = self.discriminator(features)
            loss = self.discriminator.loss(predictions, labels)
            accuracy = (torch.abs(predictions - labels) < 0.5).type(torch.FloatTensor).cuda()
            return loss, accuracy
        else:
            metrics = self.discriminator.sup_loss_on_batch(features_real, features_sim, lambda_reg=0)
            loss = metrics["wass_loss"]
            return loss


class L2Auxiliary(AuxiliaryObjective):
    def __init__(self, name, *inputs):
        super(L2Auxiliary, self).__init__(name, *inputs)
        self.loss = nn.MSELoss(size_average=True)

    def forward(self, feature_list1, feature_list2):
        assert len(feature_list1) == len(feature_list2)
        loss = self.loss(feature_list1, feature_list2)
        return loss