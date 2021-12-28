import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PVN_Stage2_RLBase(nn.Module):
    def __init__(self, map_channels=1,
                 map_struct_channels=1, map_size=32, crop_size=16, h1=8, structure_h1=8, h2=128, obs_dim=16, name="base"):
        super(PVN_Stage2_RLBase, self).__init__()

        self.map_channels = map_channels
        self.map_structure_channels = map_struct_channels
        self.map_size = map_size
        self.crop_size = crop_size
        self.name = name

        gap = int((map_size - crop_size) / 2)
        self.crop_l = gap
        self.crop_r = map_size - gap

        self.conv1 = nn.Conv2d(map_channels * 2, h1, kernel_size=3, stride=2, padding=1, bias=True)
        self.structconv1 = nn.Conv2d(self.map_structure_channels, structure_h1, kernel_size=3, stride=2, padding=1, bias=True)

        linear_in = int(((crop_size / 2) ** 2) * h1) + (8*8*structure_h1) # map channels + coverage channels + observability encoding channels
        print(f"Stage 2 linear input size: {linear_in}")
        self.linear1 = nn.Linear(linear_in, h2)
        self.linear2 = nn.Linear(h2 + linear_in, h2)

        self.avgpool = nn.AvgPool2d(4)

        self.act = nn.LeakyReLU()
        self.norm1 = nn.InstanceNorm2d(h1)
        self.covnorm1 = nn.InstanceNorm2d(structure_h1)

    def init_weights(self):
        pass

    def crop_maps(self, maps_r):
        maps_r_cropped = maps_r.inner_distribution[:, :, self.crop_l:self.crop_r, self.crop_l:self.crop_r]
        return maps_r_cropped

    def forward(self, maps_r, map_structure_r):
        maps_r_cropped = self.crop_maps(maps_r)

        # Normalize to 0-1 value range
        normmax = maps_r_cropped.max(2).values.max(2).values[:, :, None, None]
        maps_r_cropped = maps_r_cropped / (normmax + 1e-10)

        visit_oob_channels = torch.ones_like(maps_r_cropped) * maps_r.outer_prob_mass[:, :, np.newaxis, np.newaxis]
        full_maps_cropped = torch.cat([maps_r_cropped, visit_oob_channels], dim=1)

        # 64x64 -> 16x16
        uncov_r_pooled = self.avgpool(map_structure_r)

        # From 32x32 down to 16x16
        if full_maps_cropped.shape[2] == 32 and full_maps_cropped.shape[3] == 32:
            full_maps_cropped = F.avg_pool2d(full_maps_cropped, 2, stride=2)

        # From 16x16 down to 8x8
        x = self.act(self.conv1(full_maps_cropped))
        x = self.norm1(x)

        # From 16x16 down to 8x8
        c = self.act(self.structconv1(uncov_r_pooled))
        c = self.covnorm1(c)

        comb_map = torch.cat([x,c], dim=1)
        batch_size = x.shape[0]
        lin_in = comb_map.view(batch_size, -1)

        x = self.act(self.linear1(lin_in))
        x = self.act(self.linear2(torch.cat([lin_in, x], dim=1)))

        return x