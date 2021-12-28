import torch
from torch import nn as nn
import torch.nn.functional as F

from learning.inputs.common import empty_float_tensor
from learning.modules.map_transformer_base import MapTransformerBase
from learning.modules.img_to_img.img_to_features import ImgToFeatures
from learning.models.semantic_map.grid_sampler import GridSampler
from learning.models.semantic_map.pinhole_camera_inv import PinholeCameraProjectionModuleGlobal

from visualization import Presenter
from utils.simple_profiler import SimpleProfiler

from learning.inputs.pose import Pose

PROFILE = False
# Set to true to project first-person images instead of feature maps.
DEBUG_WITH_IMG = True


class ProjectToGlobalMap(MapTransformerBase):
    def __init__(self,
                 source_map_size,
                 world_size_px,
                 world_size_m,
                 img_w,
                 img_h,
                 cam_h_fov,
                 domain,
                 img_dbg=False):

        super(ProjectToGlobalMap, self).__init__(source_map_size, world_size_px, world_size_m)

        self.image_debug = img_dbg

        self.use_lang_filter = False

        # Project feature maps to the global frame
        self.map_projection = PinholeCameraProjectionModuleGlobal(
            source_map_size, world_size_px, world_size_m, img_w, img_h, cam_h_fov, domain)

        self.grid_sampler = GridSampler()

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

        self.actual_images = None

    def cuda(self, device=None):
        MapTransformerBase.cuda(self, device)
        self.map_projection.cuda(device)
        self.grid_sampler.cuda(device)
        if self.use_lang_filter:
            self.lang_filter.cuda(device)

    def init_weights(self):
        if not self.image_debug:
            self.img_to_features.init_weights()

    def reset(self):
        self.actual_images = None
        super(ProjectToGlobalMap, self).reset()

    def forward(self, images, poses, tensor_store=None, show=""):

        self.prof.tick("out")

        # self.map_projection is implemented in numpy on CPU.
        # If we give it poses on the GPU, it will transfer them to the CPU, which causes a CUDA SYNC and waits for the
        # ResNet forward pass to complete. To make use of full GPU/CPU concurrency, we move the poses to the cpu first
        poses_cpu = poses.cpu()

        # Project first-person view features on to the map in egocentric frame
        grid_maps_cpu = self.map_projection(poses_cpu)
        grid_maps = grid_maps_cpu.to(images.device)

        self.prof.tick("proj_map_and_features")
        maps = self.grid_sampler(images, grid_maps)

        # Obtain an ego-centric map mask of where we have new information
        ones_size = list(images.size())
        ones_size[1] = 1
        tmp_ones = torch.ones(ones_size).to(images.device)
        new_coverages = self.grid_sampler(tmp_ones, grid_maps)

        # Make sure that new_coverage is a 0/1 mask (grid_sampler applies bilinear interpolation)
        new_coverages = new_coverages - torch.min(new_coverages)
        new_coverages = new_coverages / (torch.max(new_coverages) + 1e-18)

        self.prof.tick("gsample")

        if show != "":
            Presenter().show_image(images.data[0, 0:3], show + "fpv_img", torch=True, scale=2, waitkey=1)

            grid_maps_np = grid_maps.data[0].cpu().numpy()

            Presenter().show_image(grid_maps_np, show + "_grid", torch=False, scale=4, waitkey=1)
            Presenter().show_image(images.data[0, 0:3].cpu(), show + "_img", torch=True, scale=1, waitkey=1)
            Presenter().show_image(maps.data[0, 0:3].cpu(), show + "_projected", torch=True, scale=6, waitkey=1)
            Presenter().show_image(new_coverages.data[0].cpu(), show + "_covg", torch=True, scale=6, waitkey=1)

        self.prof.loop()
        self.prof.print_stats(10)

        return maps, new_coverages