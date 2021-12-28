import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from visualization import Presenter


class P2DIterator:
    def __init__(self, p2d):
        self.p2d = p2d
        self.ord = -1

    def __next__(self):
        self.ord += 1
        if self.ord >= len(self.p2d.inner_distribution):
            raise StopIteration()
        inner = self.p2d.inner_distribution[self.ord]
        outer = self.p2d.outer_prob_mass[self.ord]
        return Partial2DDistribution(inner, outer)


class Partial2DDistribution(torch.nn.Module):

    def __init__(self, inner_distribution, outer_prob_mass):
        super(Partial2DDistribution, self).__init__()
        self.inner_distribution = inner_distribution
        self.outer_prob_mass = outer_prob_mass
        self.log_softmax_module = nn.LogSoftmax(dim=2)
        self.softmax_module = nn.Softmax(dim=2)

    def __getattr__(self, name):
        if name == "device":
            return self.inner_distribution.device
        else:
            return super().__getattr__(name)

    @classmethod
    def from_distribution_and_mask(cls, v_dist, cov_mask):
        # Masks a visitation distribution and creates a Partial2DDistribution
        batch_size = v_dist.shape[0]
        channels = v_dist.shape[1]
        # Normalize before masking
        for c in range(channels):
            v_dist[:, c] /= (v_dist[:, c].view([batch_size, -1]).sum(dim=1)[:, np.newaxis, np.newaxis] + 1e-10)

        # Mask distribution
        v_dist_inner_masked = v_dist * cov_mask[:v_dist.shape[0], np.newaxis, :, :]

        probs_inside = v_dist_inner_masked.view([batch_size, channels, -1]).sum(2)
        probs_outside = 1 - probs_inside
        v_dist_masked = Partial2DDistribution(v_dist_inner_masked, probs_outside)
        return v_dist_masked

    @classmethod
    def from_distribution(cls, v_dist):
        # Creates a Partial2DDistribution. If v_dist is all zeroes, assignes full probability on oob token.
        batch_size = v_dist.shape[0]
        channels = v_dist.shape[1]
        for c in range(channels):
            v_dist[:, c] /= (v_dist[:, c].view([batch_size, -1]).sum(dim=1)[:, np.newaxis, np.newaxis] + 1e-10)
        probs_inside = v_dist.view([batch_size, channels, -1]).sum(2)
        probs_outside = 1 - probs_inside
        v_dist_masked = Partial2DDistribution(v_dist, probs_outside)
        return v_dist_masked

    @classmethod
    def cross_entropy(cls, pred, target):
        """
        :param pred: Predicted raw scores (before softmax)
        :param target: Target probability distribution
        :return:
        """
        flat_pred = pred.get_full_flat_distribution()
        flat_target = target.get_full_flat_distribution()
        flat_pred_log = F.log_softmax(flat_pred, dim=-1)
        cross_ent = -flat_pred_log * flat_target
        # Sum across spatial dimension
        cross_ent = cross_ent.sum(-1)
        # Average across batches / channe;s
        cross_ent = torch.mean(cross_ent)
        return cross_ent


    @classmethod
    def stack(cls, list_of_dists, dim=0):
        inners = torch.stack([l.inner_distribution for l in list_of_dists], dim=dim)
        outers = torch.stack([l.outer_prob_mass for l in list_of_dists], dim=dim)
        out = Partial2DDistribution(inners, outers)
        return out

    @classmethod
    def cat(cls, list_of_dists, dim=0):
        inners = torch.cat([l.inner_distribution for l in list_of_dists], dim=dim)
        outers = torch.cat([l.outer_prob_mass for l in list_of_dists], dim=dim)
        out = Partial2DDistribution(inners, outers)
        return out

    def __iter__(self):
        return P2DIterator(self)

    def to(self, *args, **kwargs):
        self.inner_distribution = self.inner_distribution.to(*args, **kwargs)
        self.outer_prob_mass = self.outer_prob_mass.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        self.inner_distribution = self.inner_distribution.cuda(*args, **kwargs)
        self.outer_prob_mass = self.outer_prob_mass.cuda(*args, **kwargs)
        return self

    def cpu(self, *args, **kwargs):
        self.inner_distribution = self.inner_distribution.cpu(*args, **kwargs)
        self.outer_prob_mass = self.outer_prob_mass.cpu(*args, **kwargs)
        return self

    def get_full_flat_distribution(self):
        batch_size = self.inner_distribution.shape[0]
        num_distributions = self.inner_distribution.shape[1]
        inner_flat = self.inner_distribution.view([batch_size, num_distributions, -1])
        outer = self.outer_prob_mass.view([batch_size, num_distributions, -1])
        full_flat = torch.cat([inner_flat, outer], dim=2)
        return full_flat

    def __index__(self, *args, **kwargs):
        new_inner = self.inner_distribution.__index__(*args, **kwargs)
        new_outer = self.outer_prob_mass.__index__(*args, **kwargs)
        return Partial2DDistribution(new_inner, new_outer)

    def __getitem__(self, *args, **kwargs):
        new_inner = self.inner_distribution.__getitem__(*args, **kwargs)
        new_outer = self.outer_prob_mass.__getitem__(*args, **kwargs)
        return Partial2DDistribution(new_inner, new_outer)

    def detach(self):
        return Partial2DDistribution(self.inner_distribution.detach(), self.outer_prob_mass.detach())

    def softmax(self, logsoftmax=False):
        batch_size = self.inner_distribution.size(0)
        num_channels = self.inner_distribution.size(1)
        #assert num_channels == 2, "Must have 2 channels: visitation distribution scores and goal distribution scores"
        height = self.inner_distribution.size(2)
        width = self.inner_distribution.size(3)

        flat_inner = self.inner_distribution.view([batch_size, num_channels, -1])
        flat_outer = self.outer_prob_mass.view([batch_size, num_channels, -1])
        flat_full = torch.cat([flat_inner, flat_outer], dim=2)

        softmax_func = self.log_softmax_module if logsoftmax else self.softmax_module

        flat_softmaxed = softmax_func(flat_full)

        new_inner_distribution = flat_softmaxed[:, :, :-1].view([batch_size, num_channels, height, width])
        new_outer_prob_mass = flat_softmaxed[:, :, -1]

        return Partial2DDistribution(new_inner_distribution, new_outer_prob_mass)

    def clone(self):
        return Partial2DDistribution(self.inner_distribution.clone(), self.outer_prob_mass.clone())

    def visualize(self, idx=0, hasbatch=False, nobars=False, size=None):
        npinner = self.inner_distribution[idx] if hasbatch else self.inner_distribution if len(self.inner_distribution.shape) == 3 else self.inner_distribution[0]
        if len(npinner.shape) == 2:
            npinner = npinner[np.newaxis, :, :]
        npinner = npinner.detach().cpu().numpy().transpose((1, 2, 0))
        npouter = self.outer_prob_mass[idx] if hasbatch else self.outer_prob_mass if len(self.outer_prob_mass.shape) == 1 else self.outer_prob_mass[0]
        npouter = npouter.detach().cpu().numpy()

        # Upscale the inner distribution before drawing the bars
        if size is not None:
            # Lazy import - this class really doesn't need to depend on cv2 other than this use case
            import cv2
            scale = size // npinner.shape[0]
            npinner = cv2.resize(npinner, (scale * npinner.shape[0], scale * npinner.shape[1]), interpolation=cv2.INTER_LINEAR)

        #if len(self.inner_distribution.shape) == 4:
        #    width_axis, height_axis, channel_axis, hasbatch = 3, 2, 1, True
        #elif len(self.inner_distribution.shape) == 3 and hasbatch:
        #    width_axis, height_axis, channel_axis, hasbatch = 2, 1, None, True
        #elif len(self.inner_distribution.shape) == 3 and not hasbatch:
        #    width_axis, height_axis, channel_axis, hasbatch = 2, 1, 0, False
        #elif len(self.inner_distribution.shape) == 2:
        #    width_axis, height_axis, channel_axis, hasbatch = 1, 0, None, False
        #else:
        #    raise ValueError(f"Can't visualize distribution of shape: {self.inner_distribution.shape}")
        height_axis, width_axis, channel_axis = 0, 1, 2
        width = npinner.shape[width_axis]
        height = npinner.shape[height_axis]
        channels = npinner.shape[channel_axis] if channel_axis is not None else 1
        barwidth = int(width / 5)

        # Include 2 bars - stop and visitation
        showwidth = width# + channels * barwidth
        showheight = height
        show_img = np.zeros((showheight, showwidth, channels))

        for c in range(channels):
            npinner[:, :, c] /= (np.percentile(npinner[:, :, c], 98) + 1e-10)
        npinner = np.clip(npinner, 0, 1)

        show_img[0:height, 0:width, :] = npinner
        for c in range(channels):
            value = npouter[c]
            barheight = int(value * showheight)
            if not nobars:
                show_img[showheight - barheight:, (width - (c + 1) * barwidth):(width - c * barwidth), c] = show_img[showheight - barheight:, (width - (c + 1) * barwidth):(width - c * barwidth), c] + 0.4
            show_img = np.clip(show_img, 0.0, 1.0)
        return show_img

    def show(self, name, scale=8, waitkey=1, idx=0):
        show_img = self.visualize(idx)
        Presenter().show_image(show_img, name, scale=scale, waitkey=waitkey)
