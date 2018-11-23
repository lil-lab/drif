"""This file implements a somewhat OpenAI Gym compatible interface for training RL models"""
import math
import random
import time

import numpy as np
from transforms3d import euler

#from data_io.drone_sim import set_current_env_id, set_current_env_from_config
from data_io.env import load_path
from drones.airsim_interface.rate import Rate
from drones.droneController import DroneController
from env_config.generation.generate_random_config import make_config_with_landmark
from geometry import vec_to_yaw
from pomdp.convert_action import unnormalize_action
from pomdp.reward.imitation_reward import ImitationReward
from pomdp.reward.path_reward import FollowPathReward
from pomdp.state import DroneState

from parameters.parameter_server import get_current_parameters


class PomdpInterface:

    """Unreal must be running before this is called for now..."""
    def __init__(self, instance_id=0, cv_mode=False):
        self.instance_id = instance_id
        self.env_id = None
        self.params = get_current_parameters()["PomdpInterface"]
        step_interval = self.params["step_interval"]
        flight_height = self.params["flight_height"]
        self.scale = self.params["scale"]
        self.drone = DroneController(instance=instance_id, flight_height=flight_height)

        rate = step_interval / self.drone.clock_speed
        self.rate = Rate(rate)
        print("Adjusted rate: " + str(rate), "Step interval: ", step_interval)
        self.path = None
        self.reward = None
        self.reward_shaping = None
        self.instruction_set = None
        self.current_segment = None
        self.done = None
        self.cv_mode = cv_mode

        self.seg_start = 0
        self.seg_end = 0

    def _get_reward(self, state, drone_action):
        # If we don't have an instruction set, we can't provide a reward
        if self.instruction_set is None:
            return 0, False

        # If the agent already executed all the actions, return 0 error and done thereafter
        if self.done:
            return 0, self.done

        # Obtain the reward from the reward function
        reward, seg_complete = self.reward.get_reward(state, drone_action)
        reward += self.reward_shaping.get_reward(state, drone_action)

        # If the segment was completed, tell the reward function that we will be following the next segment
        if seg_complete:
            self.current_segment += 1
            if self.current_segment < len(self.instruction_set):
                self.seg_start = self.seg_end
                self.seg_end = self.instruction_set[self.current_segment]["end_idx"]
                self.reward.set_current_segment(self.seg_start, self.seg_end)
            # Unless this was the last segment, in which case we set this pomdp as done and return accordingly
            else:
                self.done = True
        return reward, self.done

    def set_environment(self, env_id, instruction_set=None, fast=False):
        """
        Switch the simulation to env_id. Causes the environment configuration from
        configs/configs/random_config_<env_id>.json to be loaded and landmarks arranged in the simulator
        :param env_id: integer ID
        :param instruction_set: Instruction set to follow for displaying instructions
        :param fast: Set to True to skip a delay at a risk of environment not loading before subsequent function calls
        :return:
        """
        self.env_id = env_id

        self.drone.set_current_env_id(env_id, self.instance_id)
        self.drone.reset_environment()

        # This is necessary to allow the new frame to be rendered with the new pomdp, so that the drone doesn't
        # accidentally see the old pomdp at the start
        if not fast:
            time.sleep(0.1)

        self.current_segment = 0
        self.seg_start = 0
        self.seg_end = 0

        self.path = load_path(env_id)
        if self.path is not None:
            self.reward = FollowPathReward(self.path)
            self.reward_shaping = ImitationReward(self.path)
            self.seg_end = len(self.path) - 1

        self.instruction_set = instruction_set

        if instruction_set is not None:
            self.reward.set_current_segment(self.seg_start, self.seg_end)
            if len(instruction_set) == 0:
                print("OOOPS! Instruction set of length 0!" + str(env_id))
                return

            self.seg_end = instruction_set[self.current_segment]["end_idx"]

            if self.seg_end >= len(self.path):
                print("OOOPS! For env " + str(env_id) + " segment " + str(self.current_segment) + " end oob: " + str(self.seg_end))
                self.seg_end = len(self.path) - 1

    def reset(self, seg_idx=0, landmark_pos=None, random_yaw=0):
        self.rate.reset()
        self.drone.reset_environment()
        start_pos, start_angle = self.get_start_pos(seg_idx, landmark_pos)

        if self.cv_mode:
            start_rpy = start_angle
            self.drone.teleport_3d(start_pos, start_rpy, pos_in_airsim=False)

        else:
            start_yaw = start_angle
            if self.params["randomize_init_pos"]:
                yaw_offset = float(np.random.normal(0, self.params["init_yaw_variance"], 1))
                pos_offset = np.random.normal(0, self.params["init_pos_variance"], 2)
                print("offset:", pos_offset, yaw_offset)

                start_pos = np.asarray(start_pos) + pos_offset
                start_yaw = start_yaw + yaw_offset
            self.drone.teleport_to(start_pos, start_yaw)

        self.rate.sleep(quiet=True)
        time.sleep(0.2)
        drone_state, image = self.drone.get_state()
        self.done = False

        return DroneState(image, drone_state)

    def get_start_pos(self, seg_idx=0, landmark_pos=None):

        # If we are not in CV mode, then we have a path to follow and track reward for following it closely
        if not self.cv_mode:
            if self.instruction_set:
                start_pos = self.instruction_set[seg_idx]["start_pos"]
                start_angle = self.instruction_set[seg_idx]["start_yaw"]
            else:
                start_pos = [0, 0, 0]
                start_angle = 0

        # If we are in CV mode, there is no path to be followed. Instead we are collecting images of landmarks.
        # Initialize the drone to face the position provided. TODO: Generalize this to other CV uses
        else:
            drone_angle = random.uniform(0, 2 * math.pi)
            drone_dist_mult = random.uniform(0, 1)
            drone_dist = 60 + drone_dist_mult * 300
            drone_pos_dir = np.asarray([math.cos(drone_angle), math.sin(drone_angle)])

            start_pos = landmark_pos + drone_pos_dir * drone_dist
            start_height = random.uniform(-1.5, -2.5)
            start_pos = [start_pos[0], start_pos[1], start_height]

            drone_dir = -drone_pos_dir
            start_yaw = vec_to_yaw(drone_dir)
            start_roll = 0
            start_pitch = 0
            start_angle = [start_roll, start_pitch, start_yaw]

        return start_pos, start_angle

    def reset_to_random_cv_env(self, landmark_name=None):
        config, pos_x, pos_z = make_config_with_landmark(landmark_name)
        self.drone.set_current_env_from_config(config, instance_id=self.instance_id)
        time.sleep(0.2)
        landmark_pos_2d = np.asarray([pos_x, pos_z])
        self.cv_mode = True
        return self.reset(landmark_pos=landmark_pos_2d)

    def snap_birdseye(self, fast=False, small_env=False):
        self.drone.reset_environment()
        # TODO: Check environment dimensions
        if small_env:
            pos_birdseye_as = [2.25, 2.35, -3.92]
            rpy_birdseye_as = [-1.3089, 0, 0]   # For 15 deg camera
        else:
            pos_birdseye_as = [15, 15, -25]
            rpy_birdseye_as = [-1.3089, 0, 0]   # For 15 deg camera
        self.drone.teleport_3d(pos_birdseye_as, rpy_birdseye_as, fast=fast)
        _, image = self.drone.get_state(depth=False)
        return image

    def snap_cv(self, pos, quat):
        self.drone.reset_environment()
        pos_birdseye_as = pos
        rpy_birdseye_as = euler.quat2euler(quat)
        self.drone.teleport_3d(pos_birdseye_as, rpy_birdseye_as, pos_in_airsim=True, fast=True)
        time.sleep(0.3)
        self.drone.teleport_3d(pos_birdseye_as, rpy_birdseye_as, pos_in_airsim=True, fast=True)
        time.sleep(0.3)
        _, image = self.drone.get_state(depth=False)
        return image

    def get_current_nl_command(self):
        if self.current_segment < len(self.instruction_set):
            return self.instruction_set[self.current_segment]["instruction"]
        return "FINISHED!"

    def step(self, action):
        """
        Takes an action, executes it in the simulation and returns the state, reward and done indicator
        :param action: array of length 4: [forward velocity, left velocity, yaw rate, stop probability]
        :return: DroneState object, reward (float), done (bool)
        """
        #Action
        drone_action = action[0:3]
        drone_stop = action[3]
        #print("Action: ", action)
        if drone_stop > 0.99:
            drone_action = np.array([0, 0, 0])

        self.drone.send_local_velocity_command(unnormalize_action(drone_action))
        self.rate.sleep(quiet=True)
        drone_state, image = self.drone.get_state()
        state = DroneState(image, drone_state)

        reward, done = self._get_reward(state, action)

        return state, reward, done
