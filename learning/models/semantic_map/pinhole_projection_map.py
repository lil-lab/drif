import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from pyntcloud import PyntCloud
from transforms3d import euler, quaternions


# TODO: Speed this up by computing a perspective transformation matrix
class PinholeProjector():

    def __init__(self, param_dict):

        self.map_size = param_dict["map_size"]
        self.world_size = param_dict["world_size"]
        self.map_world_size = param_dict["world_size_in_map"]
        self.res_x = param_dict["img_x"]
        self.res_y = param_dict["img_y"]
        self.use_depth = param_dict.get("use_depth") or False
        map_x = param_dict.get("cam_map_x") or 0
        map_y = param_dict.get("cam_map_y") or 0
        self.map_origin = np.asarray([map_x, map_y])
        self.cam_fov = param_dict.get("cam_fov")

    def get_focal_len(self):
        return self.res_x * 0.5 / math.tan(self.cam_fov * 0.5 * 3.14159 / 180)

    def make_camera_matrix(self):
        f = self.get_focal_len()
        return np.asarray([[f, 0, self.res_x / 2], [0, f, self.res_y / 2], [0, 0, 1]])

    def make_rotation_matrix(self, quat):
        return quaternions.quat2mat(quat)

    def make_optical_rotation_matrix(self):
        return euler.euler2mat(-1.579, 0, -1.579)

    def get_coord_grid(self, x_size, y_size, flip_x=False, flip_y=False):
        x_pixels = np.arange(0, x_size, 1)
        y_pixels = np.arange(0, y_size, 1)

        if flip_x:
            x_pixels = np.flip(x_pixels, 0)
        if flip_y:
            y_pixels = np.flip(y_pixels, 0)

        x_pixels = np.expand_dims(x_pixels, 1).repeat(y_size, 1)
        y_pixels = np.expand_dims(y_pixels, 1).repeat(x_size, 1).T

        coord_grid_2d = np.concatenate((np.expand_dims(x_pixels, 2), np.expand_dims(y_pixels, 2)), axis=2)
        z = np.ones((coord_grid_2d.shape[0], coord_grid_2d.shape[1], 1))
        coord_grid_3d = np.concatenate((coord_grid_2d, z), axis=2)
        return coord_grid_3d

    def plot_histogram(self, data):
        hist, bins = np.histogram(data)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.show()

    def get_sample(self, idx):
        with open("cam_pos_samples", "rb") as f:
            samples = pickle.load(f)

        sample = samples[idx]

        cam_pos = np.asarray(sample["pos"])
        cam_rot = np.asarray(sample["rot"])
        cam_img = sample["img"]

        cam_img = cam_img.transpose(1, 0, 2)
        cam_pos[2] = cam_pos[2] - 2.0

        return cam_img, cam_pos, cam_rot

    def plot_cloud(self, points, cam_pos=[0, 0, 0], image=None):
        if image is not None and len(image.reshape(-1)) % 3 != 0:
            return

        pts_xyz = points.squeeze()
        pts_rgb = image.reshape(-1, 3)
        pts_np = np.concatenate((pts_xyz, pts_rgb), axis=1)

        c = cam_pos
        pts_border = np.asarray(
            [
                [0.0, 0.0, 0.0, 255, 0, 0],
                [0.0, 30., 0.0, 0, 255, 0],
                [30., 0.0, 0.0, 0, 0, 255],
                [30., 30., 0.0, 255, 255, 0],
                [c[0], c[1], c[2], 0, 255, 255]
            ]
        )

        pts_np = np.concatenate((pts_np, pts_border), axis=0)

        points = pd.DataFrame(pts_np, columns=['x', 'y', 'z', 'red', 'green', 'blue'])
        cloud = PyntCloud(points)
        cloud.add_scalar_field("rgb_intensity")
        return cloud.plot()

    def world_point_to_image(self, cam_pos, cam_rot, point):
        cam_pos = cam_pos.numpy()
        cam_rot = cam_rot.numpy()

        cam_pos = cam_pos.copy()
        cam_pos[2] += DRONE_HEIGHT

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

    def get_projection_mapping_local(self, cam_pos, cam_rot, depth_im=None):

        cam_Z = cam_pos[2] + DRONE_HEIGHT

        Kinv = np.linalg.inv(self.make_camera_matrix())
        R = self.make_rotation_matrix(cam_rot)
        R_opt = self.make_optical_rotation_matrix()

        pts_grid = self.get_coord_grid(self.res_x, self.res_y, flip_x=True, flip_y=True)
        pts = pts_grid.reshape(-1, 3)
        pts = np.expand_dims(pts, axis=2)

        # Project pixel points out as rays into 3D space and rotate to the global reference frame
        rays_optical = np.matmul(np.expand_dims(Kinv, 0), pts)
        rays_camera = np.matmul(np.expand_dims(R_opt, 0), rays_optical)
        rays_global = np.matmul(np.expand_dims(R, 0), rays_camera)

        # Calculate ray lengths: the multipliers that would cause the global Z-coordinate to be 0 at the given drone height
        if depth_im is not None and self.use_depth:
            depth_im = np.squeeze(depth_im).transpose([1, 0])
            depth_im_flat = np.reshape(depth_im, (-1))
            subsampling_factor = int(math.sqrt(pts.shape[0] / depth_im_flat.shape[0]))
            pts_index_depth = pts_grid / subsampling_factor
            pts_index_depth = pts_index_depth.astype(np.uint16)[:,:,0:2]
            pts_index_depth = pts_index_depth.reshape([-1, 2])
            #pts_index_depth[:, 0] = depth_im.shape[0] - pts_index_depth[:, 0]
            pts_index_depth[:, 1] = depth_im.shape[1] - pts_index_depth[:, 1]
            pts_index_depth = pts_index_depth[:, 0] * depth_im.shape[1] + pts_index_depth[:, 1]
            pts_index_depth = np.clip(pts_index_depth, 0, depth_im_flat.shape[0] - 1)

            depth = depth_im_flat[pts_index_depth]
            depth = np.expand_dims(depth, 1)
        else:
            depth = -cam_Z / rays_global[:, 2, :]

        depth = np.clip(depth, 0, 50)  # Clip the ray length to avoid very large numbers near the horizon

        # Calculate the 3D locations of image points in the camera frame and convert to global frame
        ground_pts_optical = rays_optical * np.expand_dims(depth, 1)
        ground_pts_camera = np.matmul(np.expand_dims(R_opt, 0), ground_pts_optical).squeeze()[:, 0:2]

        scatter = ground_pts_camera.reshape((self.res_x, self.res_y, 2)) * self.map_world_size / self.world_size
        scatter += np.expand_dims(self.map_origin, 0)
        scatter = scatter.astype(int)
        scatter = np.clip(scatter, 0, self.map_size)

        scatter_2d = scatter[:, :, 1] * self.map_size + scatter[:, :, 0]
        scatter_2d = np.clip(scatter_2d, 0, self.map_size * self.map_size - 1)

        # Now need to scatter the original 2D coordinates and we will obtain the gather grid
        projection_map_x = np.full(self.map_size * self.map_size, -1)
        projection_map_y = np.full(self.map_size * self.map_size, -1)

        np.put(projection_map_x, scatter_2d.flat, pts[:, 0, :].flat)
        np.put(projection_map_y, scatter_2d.flat, pts[:, 1, :].flat)

        # Notice the negative sign! It's to do with how grid_sample defines coordinates
        projection_map_x = -(projection_map_x.reshape((self.map_size, self.map_size)) * 2 / float(self.res_x) - 1)
        projection_map_y = -(projection_map_y.reshape((self.map_size, self.map_size)) * 2 / float(self.res_y) - 1)

        projection_map = np.concatenate((np.expand_dims(projection_map_x, 2), np.expand_dims(projection_map_y, 2)), 2)
        return projection_map

    def get_projection_mapping_global(self, cam_pos, cam_rot, image=None):

        cam_Z = cam_pos[2] + DRONE_HEIGHT

        Kinv = np.linalg.inv(self.make_camera_matrix())
        R = self.make_rotation_matrix(cam_rot)
        R_opt = self.make_optical_rotation_matrix()
        pts = self.get_coord_grid(self.res_x, self.res_y, flip_x=True, flip_y=True).reshape(-1, 3)
        pts = np.expand_dims(pts, axis=2)

        # Project pixel points out as rays into 3D space and rotate them to the global reference frame
        rays_optical = np.matmul(np.expand_dims(Kinv, 0), pts)
        rays_camera = np.matmul(np.expand_dims(R_opt, 0), rays_optical)
        rays_global = np.matmul(np.expand_dims(R, 0), rays_camera)

        # Raytrace for ground: calculate the multipliers that would cause the global Z-coordinate to be 0
        # TODO: Use depth image instead
        depth = -cam_Z / rays_global[:, 2, :]
        depth = np.clip(depth, 0, 50)  # Clip the ray length to avoid very large numbers near the horizon

        # Calculate the 3D locations of image points in the camera frame and convert to global frame
        ground_pts_optical = rays_optical * np.expand_dims(depth, 1)
        ground_pts_camera = np.matmul(np.expand_dims(R_opt, 0), ground_pts_optical)
        ground_pts_camera_global_rot = np.matmul(np.expand_dims(R, 0), ground_pts_camera)
        ground_pts_global = np.expand_dims(np.expand_dims(cam_pos, axis=1), axis=0) + ground_pts_camera_global_rot

        #self.plot_cloud (ground_pts_global, cam_pos, image)

        scatter = ground_pts_global.reshape((self.res_x, self.res_y, 3)) * self.map_size / self.world_size
        scatter = scatter[:, :, 0:2].astype(int).squeeze()
        scatter = np.clip(scatter, 0, self.map_size)

        scatter_2d = scatter[:, :, 1] * self.map_size + scatter[:, :, 0]
        scatter_2d = np.clip(scatter_2d, 0, self.map_size * self.map_size - 1)

        # Now need to scatter the original 2D coordinates and we will obtain the gather grid
        semantic_gather_x = np.full(self.map_size * self.map_size, -1)
        semantic_gather_y = np.full(self.map_size * self.map_size, -1)

        np.put(semantic_gather_x, scatter_2d.flat, pts[:, 0, :].flat)
        np.put(semantic_gather_y, scatter_2d.flat, pts[:, 1, :].flat)

        # Notice the negative sign! It's to do with how grid_sample defines coordinates
        semantic_gather_x = -(semantic_gather_x.reshape((self.map_size, self.map_size)) * 2 / float(self.res_x) - 1)
        semantic_gather_y = -(semantic_gather_y.reshape((self.map_size, self.map_size)) * 2 / float(self.res_y) - 1)

        semantic_gather = np.concatenate((np.expand_dims(semantic_gather_y, 2), np.expand_dims(semantic_gather_x, 2)), 2)
        return semantic_gather