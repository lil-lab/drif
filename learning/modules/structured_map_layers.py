import spacy
import torch
import numpy as np

import torch.nn as nn


class StructuredMapLayers():

    def __init__(self, global_map_size_px):
        self.map_boundary = self.make_map_boundary(global_map_size_px)

    def make_map_boundary(self, map_size):
        boundary = torch.zeros([1, 1, map_size, map_size])
        boundary[:, :, 0, :] = 1.0
        boundary[:, :, map_size-1, :] = 1.0
        boundary[:, :, :, 0] = 1.0
        boundary[:, :, :, map_size-1] = 1.0
        return boundary

    def build(self, map_uncoverage_w, cam_poses, map_transformer_w_to_r, use_map_boundary=False):
        map_uncoverage_r, _ = map_transformer_w_to_r(map_uncoverage_w, None, cam_poses)
        if use_map_boundary:
            # Change device if necessary
            self.map_boundary = self.map_boundary.to(map_uncoverage_r.device)
            batch_size = map_uncoverage_w.shape[0]
            map_boundary_r, _ = map_transformer_w_to_r(self.map_boundary.repeat([batch_size, 1, 1, 1]), None, cam_poses)
            structured_map_info_r = torch.cat([map_uncoverage_r, map_boundary_r], dim=1)
        else:
            structured_map_info_r = map_uncoverage_r
        batch_size = map_uncoverage_w.shape[0]
        struct_info_w = torch.cat([map_uncoverage_w, self.map_boundary.repeat([batch_size, 1, 1, 1])], dim=-3)
        return structured_map_info_r, struct_info_w