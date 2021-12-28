from time import sleep
import numpy as np
from scipy.misc import imread, imsave
import skimage.draw as draw
import cv2

from data_io.instructions import get_all_instructions
from data_io.env import load_env_img
from data_io.env import load_path
from drones.aero_interface.rviz import RvizInterface
import learning.datasets.top_down_dataset as tdd
from scipy.ndimage.filters import gaussian_filter
from learning.inputs.vision import standardize_image, standardize_2d_prob_dist


from visualization import Presenter
from transformations import cf_to_img
import math

import parameters.parameter_server as P

def line_3d(start_x, start_y, start_z, end_x, end_y, end_z):
    max_len_px = max([end_x - start_x, end_y - start_y, end_z - start_z]) + 1
    coordlist=  []
    for i in range(max_len_px):
        f = float(i)/max_len_px
        x = int(start_x + (end_x - start_x) * f)
        y = int(start_y + (end_y - start_y) * f)
        z = int(start_z + (end_z - start_z) * f)
        coordlist.append([x,y,z])
    xx, yy, zz = tuple(zip(*coordlist))
    return xx, yy, zz

def plot_path_on_3d_img(img, path):
    prev_x = None
    prev_y = None
    prev_z = None
    for coord in path:
        x = int(coord[0])
        y = int(coord[1])
        z = int(coord[2])

        if prev_x is not None and prev_y is not None and prev_z is not None:
            xx, yy, zz = line_3d(prev_x, prev_y, prev_z, x, y, z)
            xx = np.clip(xx, 0, img.shape[0] - 1)
            yy = np.clip(yy, 0, img.shape[1] - 1)
            zz = np.clip(zz, 0, img.shape[2] - 1)
            img[xx, yy, zz] = 1.0
        prev_x = x
        prev_y = y
        prev_z = z
    return img

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def path_to_trajectory_fake_dynamics(path, full_time):
    time = np.zeros([len(path)])
    for i in range(len(time)):
        time[i] = sigmoid((float(i) - (len(time)/2))/len(time))
    time -= time.min()
    time /= time.max()
    time *= full_time
    trajectory = np.concatenate((path, time[:, np.newaxis]), axis=1)
    return trajectory

def get_vdist_3d(env_id, set_idx, seg_idx):
    path = load_path(env_id)
    train_i, dev_i, test_i, corpus = get_all_instructions()
    instrs = dev_i[env_id][set_idx]["instructions"][seg_idx]
    start_idx = instrs["start_idx"]
    print(instrs["instruction"])
    end_idx = instrs["end_idx"]
    seg_path = path[start_idx:end_idx]

    map_w = 32
    map_h = 32
    map_t = 32

    seg_labels = np.zeros([map_w, map_h, map_t, 3]).astype(float)
    seg_path_in_img = cf_to_img(seg_path, np.array([map_w, map_h]))
    seg_path_in_img_3d = path_to_trajectory_fake_dynamics(seg_path_in_img, map_t)
    gauss_sigma = map_w / 48

    seg_labels[:, :, :map_t-1, 0] = plot_path_on_3d_img(seg_labels[:, :, :map_t-1, 0], seg_path_in_img_3d[:-1])
    if len(seg_path_in_img_3d) > 1:
        seg_labels[:, :, map_t-1:map_t, 1] = plot_path_on_3d_img(seg_labels[:, :, map_t-1:map_t, 1], seg_path_in_img_3d[-2:])

    for x in range(seg_labels.shape[0]):
        seg_labels[x, :, :, 0] = gaussian_filter(seg_labels[x, :, :, 0], gauss_sigma)
        seg_labels[x, :, :, 1] = gaussian_filter(seg_labels[x, :, :, 1], gauss_sigma)
    for y in range(seg_labels.shape[1]):
        seg_labels[:, y, :, 0] = gaussian_filter(seg_labels[:, y, :, 0], gauss_sigma)
        seg_labels[:, y, :, 1] = gaussian_filter(seg_labels[:, y, :, 1], gauss_sigma)
    for z in range(seg_labels.shape[2]):
        seg_labels[:, :, z, 0] = gaussian_filter(seg_labels[:, :, z, 0], gauss_sigma)
        seg_labels[:, :, z, 1] = gaussian_filter(seg_labels[:, :, z, 1], gauss_sigma)

    #for i in range(seg_labels.shape[2]):
    #    Presenter().show_image(seg_labels[:, :, i, :], "l_traj", scale=4, waitkey=False)

    # TODO: Change this to proability normalization
    seg_labels[:, :, :, 0] -= seg_labels[:, :, :, 0].min()
    seg_labels[:, :, :, 0] /= seg_labels[:, :, :, 0].max()
    seg_labels[:, :, :, 1] -= seg_labels[:, :, :, 1].min()
    seg_labels[:, :, :, 1] /= seg_labels[:, :, :, 1].max()

    return seg_labels

def do_rviz_vdist():
    P.initialize_experiment()
    rviz = RvizInterface(
        voxel_topics=["env_img", "v_dist"]
    )
    while True:
        print("Publishing stuff")
        img = load_env_img(0, 512, 512, real_drone=False)
        vdist_3d = get_vdist_3d(0, 0, 0)
        vdist_3d = np.clip(vdist_3d * 3.0, 0.0, 1.0)

        rviz.publish_tensor("env_img", img, axis0_size_m=4.7, frame="/map_ned")
        rviz.publish_tensor("v_dist", vdist_3d, axis0_size_m=4.7, frame="/map_ned")
        sleep(0.1)


if __name__ == "__main__":
    do_rviz_vdist()