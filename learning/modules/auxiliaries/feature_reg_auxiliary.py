from learning.modules.auxiliary_objective_base import AuxiliaryObjective
from learning.modules.gather_2d import Gather2D

import torch
from torch import nn as nn

from learning.modules.auxiliary_objective_base import AuxiliaryObjective

from visualization import Presenter
from transformations import as_to_img

DBG = False


class FeatureRegularizationAuxiliary2D(AuxiliaryObjective):
    def __init__(self, name, world_size_px=32, kind="l1", *inputs):
        super(FeatureRegularizationAuxiliary2D, self).__init__(name, *inputs)
        self.gather_2d = Gather2D()
        self.world_size_px = world_size_px
        self.kind = kind

    def cuda(self, device=None):
        AuxiliaryObjective.cuda(self, device)
        return self

    def plot_pts(self, image, pts):
        """
        :param image: CxHxW image
        :param pts: Nx2 points - (H,W) coords in the image
        :return:
        """
        image = image.cpu().data.numpy()
        image = image.transpose((1,2,0))
        pts = pts.cpu().data.numpy()
        image[:, :, 0] = 0.0
        for pt in pts:
            image[pt[0], pt[1], 0] = 1.0

        Presenter().show_image(image[:,:,0:3], "aux_class_" + self.name, torch=False, waitkey=50, scale=8)

    def forward(self, images, lm_pos):
        images = torch.cat(images, dim=0)
        images_abv = torch.abs(images)
        loss = torch.mean(images_abv)
        return loss, 1