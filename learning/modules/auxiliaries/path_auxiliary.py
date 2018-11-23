from torch import nn as nn
import torch

from learning.modules.auxiliary_objective_base import AuxiliaryObjective
from learning.modules.crossentropy2d import CrossEntropy2d

from visualization import Presenter
DBG = False
pa2d_count = 0


class PathAuxiliary2D(AuxiliaryObjective):
    def __init__(self, name, lossfunc, *inputs):
        super(PathAuxiliary2D, self).__init__(name, *inputs)
        if lossfunc == "crossentropy":
            self.loss = CrossEntropy2d()
        else:
            self.loss = nn.MSELoss()
        self.lossfunc = lossfunc

    def cuda(self, device=None):
        AuxiliaryObjective.cuda(self, device)
        return self

    def forward(self, masks, mask_labels):
        masks = torch.cat(masks, dim=0)
        mask_labels = torch.cat(mask_labels, dim=0)

        if masks.size(1) < mask_labels.size(1):
            mask_labels = mask_labels[:, 0:masks.size(1)].contiguous()

        global pa2d_count
        if DBG and pa2d_count % 50 == 0:
            for i in range(masks.size(0)):
                Presenter().show_image(masks.data[i], "aux_path_pred", torch=True, waitkey=1, scale=4)
                Presenter().show_image(mask_labels.data[i], "aux_path_label", torch=True, waitkey=100, scale=4)
        pa2d_count += 1

        loss = self.loss(masks, mask_labels)

        # TODO: Put accuract reporting here...
        return loss, 1