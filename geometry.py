import numpy as np
import math

def yaw_to_vec(yaw):
    vec = np.zeros(3)
    vec[0] = math.cos(yaw)
    vec[1] = math.sin(yaw)
    vec[2] = 0
    return vec


def vec_to_yaw(vec):
    yaw = math.atan2(vec[1], vec[0])
    return yaw


def clip_angle(angle):
    if angle > 3.14159:
        angle -= 3.14159 * 2
    if angle < -3.14159:
        angle += 3.14159 * 2
    return angle


def pos_to_drone(drone_pos, drone_yaw, pos):
    rel_pos = pos - drone_pos
    dist = np.linalg.norm(rel_pos[0:2])
    angle = vec_to_yaw(rel_pos)
    new_angle = clip_angle(angle - drone_yaw)
    pos_wrt_drone = dist * yaw_to_vec(new_angle)
    return pos_wrt_drone


