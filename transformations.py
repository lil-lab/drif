import numpy as np
import math
from transforms3d import euler
from learning.inputs.pose import Pose
from torch.autograd import Variable
from utils.simple_profiler import SimpleProfiler
import torch


def get_affine_scale_2d(scale_vec, batch=False):
    if batch:
        out1 = np.eye(3)
        out = np.tile(out1, [len(scale_vec), 1, 1])
        out[:, 0, 0] = scale_vec[:, 0]
        out[:, 1, 1] = scale_vec[:, 1]
    else:
        out = np.eye(3)
        out[0,0] = scale_vec[0]
        out[1,1] = scale_vec[1]
    return out


def get_affine_trans_2d(trans_vec, batch=False):
    if batch:
        out1 = np.eye(3)
        out = np.tile(out1, [len(trans_vec), 1, 1])
        out[:, 0:2, 2] = trans_vec[:, 0:2]
    else:
        out = np.eye(3)
        out[0:2, 2] = trans_vec[0:2]
    return out


def get_affine_rot_2d(yaw, batch=False):
    if batch:
        out1 = np.eye(3)
        out = np.tile(out1, [len(yaw), 1, 1])
        c = np.cos(yaw)[:, 0]
        s = np.sin(yaw)[:, 0]
        out[:, 0, 0] = c
        out[:, 1, 1] = c
        out[:, 0, 1] = -s
        out[:, 1, 0] = s
    else:
        out = np.eye(3)
        c = np.cos(yaw)
        s = np.sin(yaw)
        out[0, 0] = c
        out[1, 1] = c
        out[0, 1] = -s
        out[1, 0] = s
    return out


def cf_to_img(as_coords, img_size, world_size=None, world_origin=None):
    """
    Convert an array of 2D config coordinates to an array of 2D image coordinates
    :param as_coords:
    :param img_size: (img width, img height)   width and height of the image representing the top-down view of the environment
    :param world_size: (pix width, pix height) width and height of the environment inside the image, in pixels
    :param world_origin: (x, y)                x and y coords of the upper-left corner of the environment in the image

    -----------------------
    | img   ,- w origin   |
    |      V________      |
    |      |        |     |
    |      | world  |     |
    |      |        |     |
    |      |________|     |
    |                     |
    |                     |
    -----------------------

    :return:
    """
    img_size = np.asarray(img_size)
    # Be default, assume that the image is of the entire environment
    if world_size is None:
        world_size = img_size
    # By default assume that the image is a picture of the entire environment
    if world_origin is None:
        world_origin = np.array([0.0, 0.0])

    scale = world_size / 1000
    out_coords = as_coords * scale
    out_coords[:, 1] = world_size[1] - out_coords[:, 1]
    out_coords = out_coords[:, [1,0]]
    #out_coords = world_size - out_coords
    out_coords = out_coords + world_origin
    return out_coords


def as_to_img_wrong(as_coords, img_size, world_size=None, world_origin=None):
    """
    :param img_size:
    :param as_coords:
    :return:
    """
    img_size = np.asarray(img_size)
    # Be default, assume that the image is of the entire environment
    if world_size is None:
        world_size = img_size
    # By default assume that the image is a picture of the entire environment
    if world_origin is None:
        world_origin = np.array([0.0, 0.0])

    world_size = np.asarray(world_size)
    world_origin = np.asarray(world_origin)

    scale = world_size / 30
    out_coords = as_coords * scale
    out_coords[:, 0] = world_size[0] - out_coords[:, 0]
    out_coords = out_coords + world_origin
    return out_coords


def img_to_as(img_coords, img_size, world_size=None, world_origin=None):
    img_size = np.asarray(img_size)
    # Be default, assume that the image is of the entire environment
    if world_size is None:
        world_size = img_size
    # By default assume that the image is a picture of the entire environment
    if world_origin is None:
        world_origin = np.array([0.0, 0.0])

    scale = world_size / 30

    out_coords = img_coords - world_origin
    out_coords = out_coords / scale
    out_coords = out_coords[:, [1,0]]
    return out_coords


def as_to_img(as_coords, img_size, world_size=None, world_origin=None):
    """
    :param img_size:
    :param as_coords:
    :return:
    """
    img_size = np.asarray(img_size)
    # Be default, assume that the image is of the entire environment
    if world_size is None:
        world_size = img_size
    # By default assume that the image is a picture of the entire environment
    if world_origin is None:
        world_origin = np.array([0.0, 0.0])

    if type(world_size) not in [int, float]:
        world_size = np.asarray(world_size)
    world_origin = np.asarray(world_origin)

    scale = world_size / 30

    # then flip y axis
    #out_coords[:, 1] = world_size[1] - out_coords[:, 1]
    #The above is no longer necessary because I simply rotated env images by 90 degrees so that X and Y axis align with AirSim X and Y axis
    #out_coords = world_size - out_coords
    if hasattr(as_coords, "is_cuda"):
        world_origin = torch.from_numpy(world_origin).float()
        as_coords = as_coords.float()
        if as_coords.is_cuda:
            world_origin = world_origin.cuda()
        if type(as_coords) is Variable:
            world_origin = Variable(world_origin)

    out_coords = as_coords * scale

    # first exchange x and y axes
    out_coords = out_coords[:, [1,0]]

    out_coords = out_coords + world_origin
    return out_coords


# TODO: Check whether get_landmark_locations_airsim is returning landmark locations with x and y axis swapped.
# If it is then this function is the valid one. If not, need to find inconsistency
def as_to_img_p(as_coords, img_size, world_size=None, world_origin=None):
    """
    :param img_size:
    :param as_coords:
    :return:
    """
    img_size = np.asarray(img_size)
    # Be default, assume that the image is of the entire environment
    if world_size is None:
        world_size = img_size
    # By default assume that the image is a picture of the entire environment
    if world_origin is None:
        world_origin = np.array([0.0, 0.0])

    world_size = np.asarray(world_size)
    world_origin = np.asarray(world_origin)

    scale = world_size / 30
    out_coords = as_coords * scale

    # first exchange x and y axes
    #out_coords = out_coords[:, [1,0]]
    # then flip y axis
    #out_coords[:, 1] = world_size[1] - out_coords[:, 1]
    #The above is no longer necessary because I simply rotated env images by 90 degrees so that X and Y axis align with AirSim X and Y axis
    #out_coords = world_size - out_coords
    out_coords = out_coords + world_origin
    return out_coords

# This is the biggest bottleneck in map affine transformations! Should we be precomputing this at the head?
def poses_as_to_img(as_pose, world_size_px, batch_dim=False):
    world_size_px = np.asarray(world_size_px)
    pos = as_pose.position
    rot = as_pose.orientation

    #torch.cuda.synchronize()
    #prof = SimpleProfiler(torch_sync=True, print=True)

    # Turn into numpy
    if hasattr(pos, "is_cuda"):
        pos = pos.data.cpu().numpy()
        rot = rot.data.cpu().numpy()

    if len(pos.shape) == 1:
        pos = pos[np.newaxis, :]

    #prof.tick(".")

    pos_img = as_to_img(pos[:, 0:2], world_size_px)

    #prof.tick("pos")

    yaws = []
    if batch_dim:
        #rotm = rot.copy()
        #rotm = rot
        #rotm[:, 1] = 0
        #rotm[:, 2] = 0
        for i in range(rot.shape[0]):
            # Manual quat2euler
            #mag = math.sqrt(rotm[i][0] ** 2 + rotm[i][3] ** 2)
            #rotm[i, :] /= mag
            #sign = np.sign(rotm[i][3])
            #yaw = 2*math.acos(rotm[i][0]) * sign
            roll, pitch, yaw = euler.quat2euler(rot[i])
            #print(yaw, yaw_manual, sign)
            yaws.append(yaw)
    else:
        roll, pitch, yaw = euler.quat2euler(rot)
        yaws.append(yaw)
        pos_img = pos_img[0]

    #prof.tick("rot")

    if batch_dim:
        # Add additional axis so that orientation becomes Bx1 instead of just B,
        out = Pose(pos_img, np.asarray(yaws)[:, np.newaxis])
    else:
        out = Pose(pos_img, yaws[0])

    #prof.tick("fin")
    #prof.print_stats()

    return out
