import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transforms3d import euler


class Rotation2D(nn.Module):

    def __init__(self, map_size=30, world_size=30):
        super(Rotation2D, self).__init__()
        self.map_size = map_size
        self.world_size = world_size

    def forward(self, image, cam_pos, cam_rot):

        # First build the affine 2D matrices representing the rotation around Z axis
        batch_size = image.size(0)
        affines_np = np.zeros((batch_size, 3, 3))
        for batch in range(batch_size):
            b_rot = cam_rot.data[batch].cpu().numpy()
            b_pos = cam_pos.data[batch].cpu().numpy()
            roll, pitch, yaw = euler.quat2euler(b_rot)
            c = math.cos(-yaw)
            s = math.sin(-yaw)

            pos_rel = b_pos / self.world_size
            pos_rel = pos_rel - 0.5
            pos_rel *= 2

            # Affine matrix to center the map around the drone
            affine_displacement = np.asarray([[1, 0, pos_rel[0]], [0, 1, pos_rel[1]], [0, 0, 1]])
            # Affine matrix to rotate the map to the drone's frame
            affine_rotation = np.asarray([[c, s, 0], [-s, c, 0], [0, 0, 1]])

            # Translate first, then rotate
            affines_np[batch] = np.matmul(affine_displacement, affine_rotation)

        affine_in = torch.from_numpy(affines_np[:, 0:2, :])
        affine_in = affine_in.to(image.device)

        # Build the affine grid
        grid = F.affine_grid(affine_in, torch.Size((batch_size, 1, self.map_size, self.map_size))).float()

        # Rotate the input image
        rot_img = F.grid_sample(image, grid)
        return rot_img
