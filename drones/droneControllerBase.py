import math
import numpy as np


class DroneControllerBase():

    def __init__(self):
        pass

    def _get_yaw(self):
        ...

    def _action_to_global(self, action):
        drone_yaw = self._get_yaw()
        action_global = np.zeros(3)
        # TODO: Add action[1]
        action_global[0] = action[0] * math.cos(drone_yaw)
        action_global[1] = action[0] * math.sin(drone_yaw)
        action_global[2] = action[2]
        return action_global

    def get_real_time_rate(self):
        ...

    def send_local_velocity_command(self, cmd_vel):
        ...

    def teleport_to(self, position, yaw):
        ...

    def get_state(self):
        ...

    def reset_environment(self):
        ...

    def set_current_env_id(self, env_id):
        ...