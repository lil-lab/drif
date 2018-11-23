import numpy as np
from torch import nn as nn
from torch.autograd import Variable

from learning.inputs.common import empty_float_tensor, np_to_tensor
from learning.inputs.pose import Pose
from learning.modules.affine_2d import Affine2D
from transformations import get_affine_trans_2d, get_affine_rot_2d, poses_as_to_img

from utils.simple_profiler import SimpleProfiler
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

PROFILE = False
CONCURRENT = False


class MapAffine(nn.Module):
    # TODO: Cleanup unused run_params
    def __init__(self, source_map_size, dest_map_size, world_in_map_size):
        super(MapAffine, self).__init__()
        self.is_cuda = False
        self.cuda_device = None
        self.source_map_size = source_map_size
        self.dest_map_size = dest_map_size
        self.world_in_map_size = world_in_map_size

        self.affine_2d = Affine2D()

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

        pos = np.asarray([self.source_map_size / 2, self.source_map_size / 2])
        rot = np.asarray([0])
        self.canonical_pose_src = Pose(pos, rot)

        pos = np.asarray([self.dest_map_size / 2, self.dest_map_size / 2])
        rot = np.asarray([0])
        self.canonical_pose_dst = Pose(pos, rot)

        self.executor = ProcessPoolExecutor(max_workers=10)

    def cuda(self, device=None):
        nn.Module.cuda(self, device)
        self.is_cuda = True
        self.cuda_device = device
        self.affine_2d.cuda()
        return self

    def pose_2d_to_mat_np(self, pose_2d, map_size, inv=False):
        pos = pose_2d.position
        yaw = pose_2d.orientation

        # Transform the img so that the drone's position ends up at the origin
        # TODO: Add batch support
        t1 = get_affine_trans_2d(-pos)

        # Rotate the img so that it's aligned with the drone's orientation
        yaw = -yaw
        t2 = get_affine_rot_2d(-yaw)

        # Translate the img so that it's centered around the drone
        t3 = get_affine_trans_2d([map_size / 2, map_size / 2])

        mat = np.dot(t3, np.dot(t2, t1))

        # Swap x and y axes (because of the BxCxHxW a.k.a BxCxYxX convention)
        swapmat = mat[[1,0,2], :]
        mat = swapmat[:, [1,0,2]]

        if inv:
            mat = np.linalg.inv(mat)

        return mat

    def poses_2d_to_mat_np(self, pose_2d, map_size, inv=False):
        pos = np.asarray(pose_2d.position)
        yaw = np.asarray(pose_2d.orientation)

        # Transform the img so that the drone's position ends up at the origin
        # TODO: Add batch support
        t1 = get_affine_trans_2d(-pos, batch=True)

        # Rotate the img so that it's aligned with the drone's orientation
        yaw = -yaw
        t2 = get_affine_rot_2d(-yaw, batch=True)

        # Translate the img so that it's centered around the drone
        t3 = get_affine_trans_2d(np.asarray([map_size / 2, map_size / 2]), batch=False)

        t21 = np.matmul(t2, t1)
        mat = np.matmul(t3, t21)

        # Swap x and y axes (because of the BxCxHxW a.k.a BxCxYxX convention)
        swapmat = mat[:, [1,0,2], :]
        mat = swapmat[:, :, [1,0,2]]

        if inv:
            mat = np.linalg.inv(mat)

        return mat

    def get_old_to_new_pose_mat(self, old_pose, new_pose):
        old_T_inv = self.pose_2d_to_mat_np(old_pose, self.source_map_size, inv=True)
        new_T = self.pose_2d_to_mat_np(new_pose, self.dest_map_size, inv=False)
        mat = np.dot(new_T, old_T_inv)
        #mat = new_T
        mat_t = np_to_tensor(mat, cuda=False)
        return mat_t

    def get_old_to_new_pose_matrices(self, old_pose, new_pose):
        old_T_inv = self.poses_2d_to_mat_np(old_pose, self.source_map_size, inv=True)
        new_T = self.poses_2d_to_mat_np(new_pose, self.dest_map_size, inv=False)
        mat = np.matmul(new_T, old_T_inv)
        #mat = new_T
        mat_t = np_to_tensor(mat, insert_batch_dim=False, cuda=False)
        return mat_t

    def get_affine_matrices(self, map_poses, cam_poses, batch_size):
        # Convert the pose from airsim coordinates to the image pixel coordinages
        # If the pose is None, use the canonical pose (global frame)
        if map_poses is not None:
            map_poses = map_poses.numpy() # TODO: Check if we're gonna have a list here or something
            # TODO: This is the big bottleneck. Could we precompute it in the dataloader?
            map_poses_img = poses_as_to_img(map_poses, [self.world_in_map_size, self.world_in_map_size], batch_dim=True)
        else:
            map_poses_img = self.canonical_pose_src.repeat_np(batch_size)

        if cam_poses is not None:
            cam_poses = cam_poses.numpy()
            cam_poses_img = poses_as_to_img(cam_poses, [self.world_in_map_size, self.world_in_map_size], batch_dim=True)
        else:
            cam_poses_img = self.canonical_pose_dst.repeat_np(batch_size)

        # Get the affine transformation matrix to transform the map to the new camera pose
        affines = self.get_old_to_new_pose_matrices(map_poses_img, cam_poses_img)

        return affines

    def get_affine_i(self, map_poses, cam_poses, i):
        # Convert the pose from airsim coordinates to the image pixel coordinages
        # If the pose is None, use the canonical pose (global frame)
        self.prof.tick("call")
        if map_poses is not None and map_poses[i] is not None:
            map_pose_i = map_poses[i].numpy()
            map_pose_img = poses_as_to_img(map_pose_i, [self.world_in_map_size, self.world_in_map_size])
        else:
            map_pose_img = self.canonical_pose_src

        if cam_poses is not None and cam_poses[i] is not None:
            cam_pose_i = cam_poses[i].numpy()
            cam_pose_img = poses_as_to_img(cam_pose_i, [self.world_in_map_size, self.world_in_map_size])
        else:
            cam_pose_img = self.canonical_pose_dst

        self.prof.tick("convert_pose")

        # Get the affine transformation matrix to transform the map to the new camera pose
        affine_i = self.get_old_to_new_pose_mat(map_pose_img, cam_pose_img)

        self.prof.tick("calc_affine")
        return affine_i

    def forward(self, maps, map_poses, cam_poses):
        """
        Affine transform the map from being centered around map_pose in the canonocial map frame to
        being centered around cam_pose in the canonical map frame.
        Canonical map frame is the one where the map origin aligns with the environment origin, but the env may
        or may not take up the entire map.
        :param map: map centered around the drone in map_pose
        :param map_pose: the previous drone pose in canonical map frame
        :param cam_pose: the new drone pose in canonical map frame
        :return:
        """

        # TODO: Handle the case where cam_pose is None and return a map in the canonical frame
        self.prof.tick("out")
        batch_size = maps.size(0)
        affine_matrices_cpu = Variable(empty_float_tensor([batch_size, 3, 3], False, None))

        self.prof.tick("init")

        if False:
            if CONCURRENT:
                futures = []
                for i in range(batch_size):
                    affine_i_future = self.executor.submit(self.get_affine_i, map_poses, cam_poses, i)
                    futures.append(affine_i_future)
                for i in range(batch_size):
                    affine_matrices_cpu[i] = futures[i].result()
            else:
                for i in range(batch_size):
                    affine_matrices_cpu[i] = self.get_affine_i(map_poses, cam_poses, i)
        else:
            affine_matrices_cpu = self.get_affine_matrices(map_poses, cam_poses, batch_size)

        self.prof.tick("affine_mat_and_pose")

        # Apply the affine transformation on the map
        # The affine matrices should be on CPU (if not, they'll be copied to CPU anyway!)
        maps_out = self.affine_2d(maps, affine_matrices_cpu, out_size=[self.dest_map_size, self.dest_map_size])

        self.prof.tick("affine_2d")
        self.prof.loop()
        if batch_size > 1:
            self.prof.print_stats(20)

        return maps_out