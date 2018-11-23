import torch
import numpy as np
from learning.modules.cuda_module import CudaModule


class MapBatchSelect(CudaModule):
    """
    Given a batch of B maps and poses, and a boolean mask of length B, return a batch of P maps and poses, where
    P is the number of True in the boolean mask.

    This is used to pick a subset of semantic maps for path-prediction, if we are not planning on every single timestep
    """

    def __init__(self):
        super(MapBatchSelect, self).__init__()

    def init_weights(self):
        pass

    def cuda(self, device=None):
        CudaModule.cuda(self, device)
        return self

    def forward(self, maps, map_poses, cam_poses, noisy_poses, start_poses, sent_embeds, step_enc, plan_mask=None, show=""):
        if plan_mask is None:
            return maps, map_poses, cam_poses, noisy_poses, start_poses, sent_embeds, step_enc

        mask_t = torch.Tensor(plan_mask) == True
        if self.is_cuda:
            mask_t = mask_t.cuda(self.cuda_device)

        maps_size = list(maps.size())[1:]
        select_maps = maps[mask_t[:, np.newaxis, np.newaxis, np.newaxis].expand_as(maps)].view([-1] + maps_size)
        select_sent_embeds = sent_embeds[mask_t[:, np.newaxis].expand_as(sent_embeds)].view([-1, sent_embeds.size(1)])
        select_poses = map_poses[mask_t] if map_poses is not None else None
        select_cam_poses = cam_poses[mask_t] if cam_poses is not None else None
        select_noisy_poses = noisy_poses[mask_t] if noisy_poses is not None else None
        select_start_poses = start_poses[mask_t] if start_poses is not None else None
        select_step_enc = step_enc[mask_t] if step_enc is not None else None

        #print("Selected " + str(len(select_maps)) + " maps from " + str(len(maps)))

        return select_maps, select_poses, select_cam_poses, select_noisy_poses, select_start_poses, select_sent_embeds, select_step_enc