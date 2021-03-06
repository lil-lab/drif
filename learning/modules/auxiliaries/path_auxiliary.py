from torch import nn as nn
import torch
import numpy as np

from learning.modules.auxiliary_objective_base import AuxiliaryObjective
from learning.modules.crossentropy2d import CrossEntropy2d
from learning.modules.goal_pred_criterion import GoalPredictionGoodCriterion
from learning.meters_and_metrics.goal_map_success_rate import goal_map_success_rate
from learning.modules.visitation_softmax import VisitationSoftmax

from visualization import Presenter
DBG = False
pa2d_count = 0


class PathAuxiliary2D(AuxiliaryObjective):
    def __init__(self, name, lossfunc, clip_observability, *inputs):
        super(PathAuxiliary2D, self).__init__(name, *inputs)
        if lossfunc == "crossentropy":
            self.loss = CrossEntropy2d()
        else:
            self.loss = nn.MSELoss()
        self.clip_observability = clip_observability
        self.lossfunc = lossfunc
        self.goal_location_criterion = GoalPredictionGoodCriterion()
        self.logsoftmax1d = nn.LogSoftmax(dim=1)
        self.visit_softmax = VisitationSoftmax()


    def cuda(self, device=None):
        AuxiliaryObjective.cuda(self, device)
        self.logsoftmax1d.cuda()
        self.loss.cuda()
        return self

    def forward(self, v_dist_pred, v_dist_labels, obs_masks):

        # TODO: Simplify this shit
        if self.clip_observability:
            v_dist_labels = torch.cat(v_dist_labels, dim=0)
            # It's a list of Partial2DDistribution, but actually the list should only have 1 element for the entire batch / sequence
            assert len(v_dist_pred) == 1
            v_dist_pred = v_dist_pred[0]

            batch_size = v_dist_labels.shape[0]
            obs_masks = torch.cat(obs_masks, dim=0)

            # TODO: Merge this piece of code with the one used to clip these distributions for Stage2 training
            # Renormalize goal distribution BEFORE clipping it
            sum_g_ps = v_dist_labels[:, 1, :, :].view([batch_size, -1]).sum(1)
            v_dist_labels[:, 1, :, :] = v_dist_labels[:, 1, :, :] / (sum_g_ps.view(batch_size, 1, 1) + 1e-10)
            # Renormalize visitation distributions BEFORE clipping
            sum_v_ps = v_dist_labels[:, 0, :, :].view([batch_size, -1]).sum(1)
            v_dist_labels[:, 0, :, :] = v_dist_labels[:, 0, :, :] / (sum_v_ps.view(batch_size, 1, 1) + 1e-10)

            # Clip the visitation labels according to observability masks
            v_dist_labels = v_dist_labels * obs_masks

            # Goal distributions should not be renormalized - the remaining probability should be distributed to the oob pixel
            gt_prob_goal_inside = v_dist_labels[:, 1, :, :].view([batch_size, -1]).sum(1)
            gt_prob_visit_inside = v_dist_labels[:, 0, :, :].view([batch_size, -1]).sum(1)
            gt_prob_goal_outside = 1 - gt_prob_goal_inside
            gt_prob_visit_outside = 1 - gt_prob_visit_inside

            # ------------------------------------------------------
            # Calculate goal layer loss
            # Concatenate oob_pred to pred and oob_label to label

            flat_distributions = v_dist_pred.get_full_flat_distribution()
            goal_labels_flat = v_dist_labels[:, 1:2, :, :].view(batch_size, -1)
            goal_labels_full = torch.cat([goal_labels_flat, gt_prob_goal_outside[:, np.newaxis]], dim=1)
            visit_labels_flat = v_dist_labels[:, 0:1, :, :].view(batch_size, -1)
            visit_labels_full = torch.cat([visit_labels_flat, gt_prob_visit_outside[:, np.newaxis]], dim=1)

            pred_visit_logsoftmax_scores = self.logsoftmax1d(flat_distributions[:, 0])
            pred_goal_logsoftmax_scores = self.logsoftmax1d(flat_distributions[:, 1])

            # Calculate losses
            xg = -goal_labels_full * pred_goal_logsoftmax_scores
            xv = -visit_labels_full * pred_visit_logsoftmax_scores
            # Sum over spatial dimensions:
            xg = xg.sum(1)
            xv = xv.sum(1)
            # Average over channels and batches
            goal_loss = torch.mean(xg)
            visit_loss = torch.mean(xv)

            # ------------------------------------------------------
            #goal_loss = self.loss(v_dist_pred[:, 1:2, :, :], v_dist_labels[:, 1:2, :, :], goal_not_in_map_score_preds, prob_goal_not_in_map_labels)

            loss = visit_loss + goal_loss

            # We're done! Calculating metrics now...
            # ------------------------------------------------------

            metrics = {}
            pred_probs_flat = torch.exp(pred_goal_logsoftmax_scores)
            pred_prob_goal_outside = pred_probs_flat[:, -1]
            pred_prob_goal_inside = 1 - pred_prob_goal_outside
            pred_goal_inside = (pred_prob_goal_inside > 0.5).long()

            gt_goal_inside = (gt_prob_goal_inside > 0.5).long()
            gt_goal_outside = (1 - gt_goal_inside)

            # Inside/Outside accuracy
            goal_insideness_correct = (pred_goal_inside == gt_goal_inside).long()
            num_goal_insideness_correct = goal_insideness_correct.sum()
            goal_inside_accuracy = num_goal_insideness_correct / (batch_size + 1e-9)
            metrics["goal_inside_outside_accuracy"] = goal_inside_accuracy

            # Inside recall
            goal_inside_recalled = (goal_insideness_correct * gt_goal_inside).sum().float()
            goal_inside_total = (gt_goal_inside.sum()).float()
            goal_inside_recall = goal_inside_recalled / (goal_inside_total + 1e-9)
            if goal_inside_recall.detach().item() > 1e-9:
                metrics["goal_inside_recall"] = goal_inside_recall

            # Outside recall
            goal_outside_recalled = (goal_insideness_correct * gt_goal_outside).sum().float()
            goal_outside_total = (gt_goal_outside.sum()).float()
            goal_outside_recall = goal_outside_recalled / (goal_outside_total + 1e-9)
            if goal_outside_recall.detach().item() > 1e-9:
                metrics["goal_outside_recall"] = goal_outside_recall

            # Location success rate given ground truth goal is inside
            pred_goal_maps_where_goal_is_inside = v_dist_pred.inner_distribution[gt_goal_inside.byte(), 1, :, :]
            gt_goal_maps_where_goal_is_inside = v_dist_labels[gt_goal_inside.byte(), 1, :, :]

            if gt_goal_maps_where_goal_is_inside.shape[0] > 0:
                goal_success_rate = goal_map_success_rate(pred_goal_maps_where_goal_is_inside, gt_goal_maps_where_goal_is_inside)
                metrics["goal_success_rate_when_goal_inside"] = goal_success_rate
                metrics["goal_inside"] = 1
            else:
                metrics["goal_inside"] = 0

            count = 1
        else:
            # TODO: Deal with these being Partial2DDistributions
            v_dist_pred = torch.cat([v.inner_distribution for v in v_dist_pred], dim=0)
            v_dist_labels = torch.cat(v_dist_labels, dim=0)
            loss = self.loss(v_dist_pred, v_dist_labels)
            metrics = {}
            count = 1

        global pa2d_count
        if DBG and pa2d_count % 10 == 0:
            for i in range(v_dist_pred.size(0)):
                Presenter().show_image(v_dist_pred.data[i, 0], "aux_path_pred", waitkey=1, scale=8)
                Presenter().show_image(v_dist_labels.data[i, 0], "aux_path_label", waitkey=1, scale=8)
        pa2d_count += 1

        return loss, metrics, count
