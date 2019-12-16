import numpy as np
import math
from transforms3d import euler

import parameters.parameter_server as P

DEPTH_SCALE = 255 / 30.0


# Class that converts between config units and unreal units
# Config units are consistent with the unity environments on which the data was collected
# Unreal units are consistent with the unreal pomdp, in centimeters
# AirSim unitys are same as unreal, except expressed in meters_and_metrics.
class UnrealUnits:
    def __init__(self):
        self.scale = self.get_scale()

    def get_scale(self):
        return P.get_current_parameters()["Units"]["scale"]

    def get_config_scale(self):
        return np.asarray(P.get_current_parameters()["Units"]["config_scale"])

    def get_config_size(self):
        return np.asarray(P.get_current_parameters()["Units"]["config_size"])

    def get_config_origin(self):
        return np.asarray(P.get_current_parameters()["Units"]["config_origin"])

    def get_env_size(self):
        return np.asarray(P.get_current_parameters()["Units"]["env_size"])

    def get_env_origin(self):
        return np.asarray(P.get_current_parameters()["Units"]["env_origin"])

    #3D Velocity and positions to and from UE
    def pos3d_to_ue(self, point):
        point = np.asarray(point)
        normalizing_factor = self.get_config_scale() / self.get_config_size()
        point_normalized = (point - self.get_config_origin()) * normalizing_factor
        point_ue = (point_normalized * self.get_env_size()) + self.get_env_origin()
        return point_ue * self.scale

    def pos3d_from_ue(self, point_ue):
        point_ue = np.asarray(point_ue)
        point_normalized = (point_ue / self.scale - self.get_env_origin()) / self.get_env_size()
        point_config = point_normalized * (self.get_config_size() / self.get_config_scale()) + self.get_config_origin()
        return point_config

    def vel3d_to_ue(self, vel):
        vel = np.asarray(vel)
        vel_normalized = vel * self.get_config_scale() / self.get_config_size()
        vel_ue = vel_normalized * self.get_env_size()
        return vel_ue * self.scale

    def vel3d_from_ue(self, vel_ue):
        vel_ue = np.asarray(vel_ue)
        vel_normalized = vel_ue / (self.scale * self.get_env_size())
        vel_config = vel_normalized * self.get_config_size() / self.get_config_scale()
        return vel_config


    #2D Velocity and positions to and from UE
    def pos2d_to_ue(self, point):
        temp = np.zeros(3)
        temp[0:2] = point
        res = self.pos3d_to_ue(temp)[:2]
        return res

    def pos2d_from_ue(self, point):
        temp = np.zeros(3)
        temp[0:2] = point
        return self.pos3d_from_ue(temp)[:2]

    def vel2d_to_ue(self, point):
        temp = np.zeros(3)
        temp[0:2] = point
        return self.vel3d_to_ue(temp)[:2]

    def vel2d_from_ue(self, point):
        temp = np.zeros(3)
        temp[0:2] = point
        return self.vel3d_from_ue(temp)[:2]


    #3D Velocity and positions to and from AirSim
    def pos3d_to_as(self, point):
        return self.pos3d_to_ue(point) / 100

    def pos3d_from_as(self, point):
        point = np.asarray(point)
        return self.pos3d_from_ue(point * 100)

    def pos2d_to_as(self, point):
        # TODO: Double check if this is correct. It seems that UE4 x-axis might be AirSim's y axis.
        return self.pos2d_to_ue(point) / 100

    def pos2d_from_as(self, point):
        point = np.asarray(point)
        return self.pos2d_from_ue(point * 100)


    #2D Velocity and positions to and from AirSim
    def vel3d_to_as(self, point):
        return self.vel3d_to_ue(point) / 100

    def vel3d_from_as(self, point):
        point = np.asarray(point)
        return self.vel3d_from_ue(point * 100)

    def vel2d_to_as(self, point):
        return self.vel2d_to_ue(point) / 100

    def vel2d_from_as(self, point):
        point = np.asarray(point)
        return self.vel2d_from_ue(point * 100)

    #Angle conversion
    def yaw_rate_to_as(self, yaw_rate):
        return -yaw_rate

    def yaw_to_as(self, yaw):
        tmp_vec = np.asarray([math.cos(yaw), math.sin(yaw)])
        as_vec = self.vel2d_to_as(tmp_vec)
        as_yaw = math.atan2(as_vec[1], as_vec[0])
        return as_yaw

    def yaw_from_as(self, yaw):
        tmp_vec = np.asarray([math.cos(yaw), math.sin(yaw)])
        config_vec = self.vel2d_from_as(tmp_vec)
        config_yaw = math.atan2(config_vec[1], config_vec[0])
        return config_yaw

    def euler_from_as(self, roll, pitch, yaw):
        rot_vec_as, theta = euler.euler2axangle(roll, pitch, yaw)
        rot_vec = self.vel3d_from_as(rot_vec_as)
        r, p, y = euler.axangle2euler(rot_vec, theta)
        return r, p, y

    #Shorthand height conversion
    def height_to_as(self, height):
        return self.pos3d_to_as([0, 0, height])[2]

    def height_from_as(self, height):
        return self.pos3d_from_as([0, 0, height])[2]