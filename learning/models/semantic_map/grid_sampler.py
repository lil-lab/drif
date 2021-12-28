import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from learning.models.semantic_map.pinhole_projection_map import PinholeProjector
#from learning.models.semantic_map.sm_params import MAP_SIZE, MAP_WORLD_SIZE, WORLD_SIZE, IMAGE_X, IMAGE_Y, USE_DEPTH


class GridSampler(nn.Module):
    def __init__(self):
        super(GridSampler, self).__init__()

    def forward(self, image, gather_maps):
        # Semantic map
        #image = image.transpose(2, 3)   # Transpose the width and height before projection
        semantic_maps_out = F.grid_sample(image, gather_maps, align_corners=False)
        return semantic_maps_out


#TODO: Deprecate this and replace with GridSampler
class PinholeProjectorPytorch(nn.Module):

    def __init__(self, map_size, map_world_size, world_size, img_x, img_y, use_depth=False):
        super(PinholeProjectorPytorch, self).__init__()
        self.map_size = map_size
        self.map_world_size = map_world_size
        self.world_size = world_size
        self.projector = PinholeProjector(
            map_size=map_size,
            map_world_size=map_world_size,
            world_size=world_size,
            img_x=img_x,
            img_y=img_y,
            use_depth=use_depth)

    def get_gather_maps(self, depth_images_np, pos, rot):
        batch_size = pos.size(0)

        gather_maps = np.zeros((batch_size, self.map_size, self.map_size, 2))
        for batch in range(batch_size):
            b_pos = pos.data[batch].cpu().numpy()
            b_rot = rot.data[batch].cpu().numpy()
            b_depth_image = depth_images_np[batch]#.cpu().numpy()
            gather_maps[batch] = self.projector.get_projection_mapping_local(b_pos, b_rot, depth_im=b_depth_image)

        # ----
        gather_maps = torch.from_numpy(gather_maps).float()
        gather_maps = Variable(gather_maps)
        return gather_maps

    def forward(self, image, gather_maps):
        # Semantic map
        #image = image.transpose(2, 3)   # Transpose the width and height before projection
        semantic_maps_out = F.grid_sample(image, gather_maps, align_corners=False)
        return semantic_maps_out