import math
import numpy as np
import torch.nn as nn
from transforms3d import euler, quaternions, affines
from transformations import pos_px_to_m

import torch
from learning.inputs.common import empty_float_tensor

# This should be 0.15 in the simulator and
H_OFFSET_REAL = -0.15
H_OFFSET_SIM = 0.0


class PinholeCameraProjection():

    def __init__(self,
                 map_size_px,
                 world_size_px,
                 world_size_m,
                 img_x,
                 img_y,
                 cam_fov,
                 domain,
                 use_depth=False,
                 start_height_offset=0.0):

        # TODO: For 30x30m AirSim simulator start_height_offset should be -2.0
        self.map_size_px = map_size_px
        self.world_size_m = world_size_m
        self.map_world_size_px = world_size_px

        self.res_x = img_x
        self.res_y = img_y
        self.use_depth = use_depth
        self.cam_fov = cam_fov
        self.h_offset = H_OFFSET_SIM if domain == "sim" else H_OFFSET_REAL

        self.cache = {}

    def get_focal_len(self):
        return self.res_x * 0.5 / math.tan(self.cam_fov * 0.5 * 3.14159 / 180)

    def make_camera_matrix(self):
        f_x = self.get_focal_len()
        f_y = f_x
        return np.asarray([[f_x, 0, self.res_x / 2], [0, f_y, self.res_y / 2], [0, 0, 1]])

    def make_rotation_matrix(self, quat):
        return quaternions.quat2mat(quat)

    def make_optical_rotation_matrix(self):
        return euler.euler2mat(-1.579, 0, -1.579)

    def make_world_to_camera_mat(self, cam_pos, cam_rot):
        rot_mat = quaternions.quat2mat(cam_rot)
        mat = affines.compose(cam_pos, rot_mat, [1.0, 1.0, 1.0])
        return mat

    def get_coord_grid(self, x_size, y_size, flip_x=True, flip_y=True, use_3d=False):
        key = str(x_size) + "." + str(y_size)
        self.cache = {}
        if key in self.cache:
            return self.cache[key]
        else:
            x_pixels = np.arange(0, x_size, 1)
            y_pixels = np.arange(0, y_size, 1)

            if flip_x:
                x_pixels = np.flip(x_pixels, 0)
            if flip_y:
                y_pixels = np.flip(y_pixels, 0)

            x_pixels = np.expand_dims(x_pixels, 1).repeat(y_size, 1)
            y_pixels = np.expand_dims(y_pixels, 1).repeat(x_size, 1).T

            coord_grid_2d = np.concatenate((np.expand_dims(x_pixels, 2), np.expand_dims(y_pixels, 2)), axis=2)

            if use_3d is not False:
                z = np.zeros((coord_grid_2d.shape[0], coord_grid_2d.shape[1], 1))
                coord_grid_2d = np.concatenate((coord_grid_2d, z), axis=2)

            zhom = np.ones((coord_grid_2d.shape[0], coord_grid_2d.shape[1], 1))
            coord_grid_homo = np.concatenate((coord_grid_2d, zhom), axis=2)
            self.cache[key] = coord_grid_homo
            return coord_grid_homo

    def get_coord_grid_fast(self, x_size, y_size):
        x_pixels = np.arange(x_size-1, -1, -1)
        y_pixels = np.arange(y_size-1, -1, -1)
        vx, vy = np.meshgrid(x_pixels, y_pixels)
        vx = vx[:, :, np.newaxis]
        vy = vy[:, :, np.newaxis]
        z = np.ones(vx.shape)
        return np.concatenate((vy, vx, z), axis=2)

    def world_point_to_image(self, cam_pos, cam_rot, point):
        if hasattr(cam_pos, "cuda"):
            cam_pos = cam_pos.detach().cpu().numpy()
            cam_rot = cam_rot.detach().cpu().numpy()

        cam_pos = cam_pos.copy()
        cam_pos[2] += self.h_offset

        K = self.make_camera_matrix()
        R = self.make_rotation_matrix(cam_rot).T
        R_opt = self.make_optical_rotation_matrix().T

        point_in_cam_pos = point - cam_pos              # Point in global frame centered around camera position
        point_in_cam = np.dot(R, point_in_cam_pos)      # Point in camera frame
        point_in_cam_opt = np.dot(R_opt, point_in_cam)  # Point in camera's optical frame

        # This landmark is behind the drone not in front of it
        if point_in_cam_opt[2] < 0:
            return None, point_in_cam_opt, "point behind"

        image_point = np.dot(K, point_in_cam_opt)       # Point in the image in pixels coordinates
        image_point_in_pixels = image_point / image_point[2]

        # Landmark is in front of drone but out of image bounds
        if image_point_in_pixels[0] < 0 or image_point_in_pixels[0] > self.res_x or \
           image_point_in_pixels[1] < 0 or image_point_in_pixels[1] > self.res_y:
            return None, point_in_cam_opt, "point oob"

        # Make sure coordinates are consistent with images. For some reason this is necessary...
        image_point_in_pixels_out = np.asarray(
            [self.res_y - image_point_in_pixels[1],
             self.res_x - image_point_in_pixels[0]]
        )

        return image_point_in_pixels_out, point_in_cam_opt, "point in camera"

    def get_world_coord_grid(self):
        coord_grid = self.get_coord_grid(self.map_size_px, self.map_size_px, flip_x=False, flip_y=False, use_3d=True).reshape(-1, 4)
        coord_grid[:, 0:2] = pos_px_to_m(coord_grid[:, 0:2], self.map_size_px, self.world_size_m, self.map_world_size_px)
        return coord_grid

    def get_projection_mapping(self, cam_pos, cam_rot, local_frame=False, range1=True):
        """
        For each pixel in the global map, compute the location of that pixel in the image
        :param cam_pos: camera position
        :param cam_rot: camera orientation
        :param image:
        :return:
        """

        cam_pos = cam_pos.copy()
        cam_pos[2] += self.h_offset

        K = self.make_camera_matrix()
        R_opt = self.make_optical_rotation_matrix()
        T_opt = affines.compose([0, 0, 0], R_opt, [1.0, 1.0, 1.0])
        T_opt_inv = np.linalg.inv(T_opt)
        T = self.make_world_to_camera_mat(cam_pos, cam_rot)
        Tinv = np.linalg.inv(T)

        # Get the map position encodings (MxMx3)
        pts_w = self.get_world_coord_grid()[..., np.newaxis]

        # Get the coordinates in camera frame:
        if not local_frame:
            # If we're using a global map frame, transform the map coordinates into the camera frame
            pts_cam = np.matmul(Tinv[np.newaxis, ...], pts_w)
        else:
            # If we're using local frame, camera is centered in the map, but pitch must still be taken into account!
            # TODO: Fix this and add pitch
            pts_cam = pts_w
            pts_cam[:, 0:2] = pts_cam[:, 0:2] - self.map_world_size_px / 2

        # Get the coordinates in optical frame
        pts_opt = np.matmul(T_opt_inv[np.newaxis, ...], pts_cam)

        # Get the 3D coordinates of the map pixels in the image frame:
        pts_img = np.matmul(K[np.newaxis, ...], pts_opt[:, 0:3, :])

        # Convert to homogeneous (image-plane) coordinates
        valid_z = pts_img[:, 2:3, :] > 0

        pts_img = pts_img / (pts_img[:, 2:3] + 1e-9)
        #pts_img[:, 0] = pts_img[:, 0] / (pts_img[:, 2] + 1e-9)
        #pts_img[:, 1] = pts_img[:, 1] / (pts_img[:, 2] + 1e-9)

        # Mask out all the map elements that don't project on the image
        valid_y1 = pts_img[:, 0:1, :] > 0
        valid_y2 = pts_img[:, 0:1, :] < self.res_x
        valid_x1 = pts_img[:, 1:2, :] > 0
        valid_x2 = pts_img[:, 1:2, :] < self.res_y

        # Throw away the homogeneous Z coordinate
        pts_img = pts_img[:, 0:2]

        valid = valid_y1 * valid_y2 * valid_x1 * valid_x2 * valid_z

        # PyTorch takes projection mappings in -1 to 1 range:
        if range1:
            pts_img[:, 0] = (-pts_img[:, 0] + self.res_x / 2) / (self.res_x / 2)
            pts_img[:, 1] = (-pts_img[:, 1] + self.res_y / 2) / (self.res_y / 2)

            # Make sure the invalid points are out of range
            pts_img = pts_img * valid + 2 * np.ones_like(pts_img) * (1 - valid)
        else:
            pts_img = pts_img * valid

        # Remove the extra 1-length dimension
        pts_img = pts_img.squeeze()

        # Reshape into the 2D map representation
        pts_img = np.reshape(pts_img, [self.map_size_px, self.map_size_px, 2])

        return pts_img


class PinholeCameraProjectionModuleGlobal(nn.Module):

    def __init__(self,
                 map_size,
                 world_in_map_size,
                 world_size,
                 img_w, img_h,
                 cam_fov,
                 domain):
        super(PinholeCameraProjectionModuleGlobal, self).__init__()

        self.map_size = map_size
        self.projector = PinholeCameraProjection(map_size, world_in_map_size, world_size, img_w, img_h, cam_fov, domain, False)

    def forward(self, cam_pose):
        batch_size = len(cam_pose)
        out_cpu = empty_float_tensor([batch_size, self.map_size, self.map_size, 2])

        # TODO: parallel for loop this
        for i in range(batch_size):
            mapping_i_np = self.projector.get_projection_mapping(cam_pose[i].position.cpu().data.numpy(), cam_pose[i].orientation.cpu().data.numpy(), range1=True)
            mapping_i = torch.from_numpy(mapping_i_np).float()
            out_cpu[i, :, :, :] = mapping_i

        return out_cpu
