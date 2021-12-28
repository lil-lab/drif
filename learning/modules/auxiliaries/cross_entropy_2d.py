import torch
from torch import nn as nn
import torch.nn.functional as F

from learning.modules.auxiliary_objective_base import AuxiliaryObjective
from learning.modules.crossentropy2d import CrossEntropy2d
from learning.modules.goal_pred_criterion import GoalPredictionGoodCriterion
from learning.inputs.partial_2d_distribution import Partial2DDistribution
DBG = False
pa2d_count = 0


class CrossEntropy2DAuxiliary(AuxiliaryObjective):
    def __init__(self, name, *inputs):
        super(CrossEntropy2DAuxiliary, self).__init__(name, *inputs)
        self.loss = CrossEntropy2d()
        self.goal_location_criterion = GoalPredictionGoodCriterion()

    def cuda(self, device=None):
        AuxiliaryObjective.cuda(self, device)
        return self

    def forward(self, pred_dist, labels):
        if isinstance(pred_dist[0], Partial2DDistribution):
            pred_t = pred_dist[0]
            labels_t = labels[0]
            pred_t_flat = pred_t.get_full_flat_distribution()
            label_t_flat = labels_t.get_full_flat_distribution()
            label_t_flat = label_t_flat.to(pred_t_flat.device)
            x = - label_t_flat * F.log_softmax(pred_t_flat, dim=2)
            x = x.sum(2)
            # Average over channels and batches
            loss = torch.mean(x)
            #loss = self.loss(pred_t_flat, label_t_flat)
            #pred_inner = pred_t.inner_distribution
            #pred_outer = pred_t.outer_prob_mass
            #labels_inner = labels_t.inner_distribution
            #labels_outer = labels_t.outer_prob_mass
            #loss = self.loss(pred_inner, labels_inner, oob_pred=pred_outer, oob_label=labels_outer)
        else:
            pred_t = torch.cat(pred_dist, dim=0)
            labels_t = torch.cat(labels, dim=0)
            loss = self.loss(pred_t, labels_t)

        metrics = {}
        count = 1
        return loss, metrics, count
