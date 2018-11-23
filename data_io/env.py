import json
import os

import numpy as np
from scipy.misc import imread, imsave
import skimage.transform as transform
from transforms3d import quaternions, euler

from data_io import paths
from data_io.helpers import load_json, save_json


def load_env_config(env_id):
    with open(paths.get_env_config_path(env_id)) as fp:
        config = json.load(fp)
    return config


def get_available_env_ids():
    conf_dir = paths.get_env_config_dir()
    jsons = os.listdir(conf_dir)
    jsons = [j for j in jsons if j.endswith(".json")]
    # extract ID from random_config_ID.json where ID is an integer
    ids = [int(j.split("_")[2].split(".")[0]) for j in jsons]
    return list(sorted(ids))


def get_list_of_sample_ids(env_id):
    poses_dir = paths.get_poses_dir(env_id)
    jsons = os.listdir(poses_dir)
    jsons = [j for j in jsons if j.endswith(".json")]
    pose_ids = [int(j.split("_")[1].split(".")[0]) for j in jsons]
    return list(sorted(pose_ids))


def load_template(env_id):
    try:
        with open(paths.get_template_path(env_id)) as fp:
            template_dict = json.load(fp)
            return template_dict
    except Exception as e:
        return None


def load_instructions(env_id):
    # TODO: Get this from annotations
    with open(paths.get_instructions_path(env_id)) as fp:
        instructions = fp.read(10000)
        return instructions


def load_path(env_id, anno=True):
    anno_curve_path = paths.get_anno_curve_path(env_id)
    if os.path.isfile(anno_curve_path) and anno:
        path = load_json(anno_curve_path)
    else:
        path = load_json(paths.get_curve_path(env_id))
    if path is None:
        print("Ground truth path not found for env: " + str(env_id))
        return path
    x_arr = path['x_array']
    y_arr = path['z_array']
    path = np.asarray(list(zip(x_arr, y_arr)))
    return path


def pose_ros_enu_to_airsim_ned(pos, quat):
    pos_as = pos.copy()
    quat_out = quat.copy()

    # ROS xyzw to Transforms3D/AirSim wxyz
    quat_out[1:4] = quat[0:3]
    quat_out[0] = quat[3]
    quat = quat_out.copy()

    R_map_enu_to_drn_enu = quaternions.quat2mat(quat)

    rot_x, rot_y, rot_z = euler.mat2euler(R_map_enu_to_drn_enu)
    R_map_enu_to_drn_enu = euler.euler2mat(rot_x, -rot_y, rot_z)

    R_ros_enu_to_as_ned = np.asarray([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])
    R_as_ned_to_ros_enu = R_ros_enu_to_as_ned.T

    R_map_ned_to_drn_enu = np.dot(R_map_enu_to_drn_enu, R_as_ned_to_ros_enu)
    R_map_ned_to_drn_ned = np.dot(R_ros_enu_to_as_ned, R_map_ned_to_drn_enu)
    R_drn_ned_to_drn_as = np.asarray([
        [0, -1, 0],  # 0, -1, 0
        [1, 0, 0],  # 1, 0, 0
        [0, 0, 1]
    ])
    R_map_ned_to_drn_as = np.dot(R_drn_ned_to_drn_as, R_map_ned_to_drn_ned)

    quat_as = quaternions.mat2quat(R_map_ned_to_drn_as)

    as_rot_x, as_rot_y, as_rot_z = euler.quat2euler(quat_as)
    print("AirSim euler: ", as_rot_x*180/np.pi, as_rot_y*180/np.pi, as_rot_z*180/np.pi)

    # ENU to NED Position
    # The drone in AirSim starts at 1m height, so we have to subtract back 1m. In practice a bit less to account
    # for the fact that the real-world camera is mounted a bit higher.
    pos_as[2] = -pos[2] + 0.8
    pos_as[1] = pos[0]
    pos_as[0] = pos[1]

    #quat_as = euler.euler2quat(0, 0, 0)
    #pos_as = [0, 0, 0]

    return pos_as, quat_as


def load_real_drone_pose(env_id, pose_id):
    """
    Loads drone pose saved from ROS and converts to AirSim convention
    AirSim convention is NED, with quaternions in w,x,y,z order.
    ROS convention is ENU with quaternions in x,y,z,w order.
    :param env_id:
    :param pose_id:
    :return:
    """
    path = paths.get_pose_path(env_id, pose_id)
    with open(path) as fp:
        pose_json = json.load(fp)
    drone_pose = pose_json["drone"]
    # The real drone pose is expressed in the map ENU frame
    # The drone's axis are: x-forward, y-right, z-up
    # The simulator drone pose is expressed in the map NED frame
    # The sim drone's axis are x-forward, y-left, z-down
    # TODO: Convert properly

    pos = np.asarray(drone_pose["position"])
    quat = np.asarray(drone_pose["orientation"])

    pos, quat = pose_ros_enu_to_airsim_ned(pos, quat)

    return pos, quat


def load_real_drone_image(env_id, pose_id):
    """
    :param env_id:
    :param pose_id:
    :return:
    """
    path = paths.get_real_img_path(env_id, pose_id)
    img = imread(path)
    return img


def save_sim_drone_image(env_id, pose_id, img):
    """
    :param env_id:
    :param pose_id:
    :return:
    """
    path = paths.get_sim_img_path(env_id, pose_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("imsave: ", path)
    imsave(path, img)


def save_anno_path(env_id, path):
    anno_curve_path = paths.get_anno_curve_path(env_id)
    path_json = {
        'x_array': list(path[:, 0]),
        'z_array': list(path[:, 1])
    }
    save_json(path_json, anno_curve_path)
    return path


def load_env_img(env_id, width=None, height=None):
    img = imread(paths.get_env_image_path(env_id))
    if width is not None:
        img = transform.resize(
            img, [width, height], mode="constant")

    return np.array(img)


def save_env_split(dict_of_lists):
    path = paths.get_env_split_path()
    save_json(dict_of_lists, path)


def load_env_split():
    path = paths.get_env_split_path()
    split = load_json(path)
    return split