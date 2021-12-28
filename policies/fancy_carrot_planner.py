import math

import numpy as np

import pomdp.convert_action as DC
from geometry import vec_to_yaw, clip_angle, pos_to_drone
from policies.abstract_policy import AbstractPolicy
from parameters.parameter_server import get_current_parameters


SETTINGS = None


class FancyCarrotPlanner(AbstractPolicy):

    def __init__(self, path=None, max_deviation=200):
        self.path = None
        self.end_idx = 0

        self.params = get_current_parameters()["FancyCarrotPlanner"]
        self.dynamics_params = get_current_parameters()["Dynamics"]

        self.current_step = 0
        self.current_lookahead = self.params["LOOKAHEAD"]
        self.current_vel_lookahed = self.params["VEL_LOOKAHEAD"]
        self.finished = False
        self.discrete = False
        self.control_step = 0
        self.max_deviation = max_deviation

        self.last_sanity_pos = np.zeros(3)
        self.set_path(path)

    def set_path(self, path):
        self.path = np.asarray(path)
        if path is not None:
            self.end_idx = len(path) - 2

    def clip_vel(self, action):
        # Never go backwards
        if action[0] > self.dynamics_params["MAX_VEL_X"]:
            action[0] = self.dynamics_params["MAX_VEL_X"]
        if action[0] < 0:
            action[0] = 0

        if action[2] > self.dynamics_params["MAX_VEL_Theta"]:
            action[2] = self.dynamics_params["MAX_VEL_Theta"]
        if action[2] < -self.dynamics_params["MAX_VEL_Theta"]:
            action[2] = -self.dynamics_params["MAX_VEL_Theta"]
        return action

    def get_action_target_yaw(self, target_yaw, current_yaw):
        diff_yaw = target_yaw - current_yaw
        vel_yaw = diff_yaw * self.params["K_Yaw"]
        action = np.asarray([0, 0, vel_yaw])
        action = self.clip_vel(action)
        return action

    def set_current_segment(self, start_idx, end_idx):
        #print ("Starting segment: [" + str(start_idx) + ", " + str(end_idx) + "]")
        self.end_idx = end_idx
        self.current_step = start_idx
        self.current_lookahead = self.params["LOOKAHEAD"]
        self.current_vel_lookahed = self.params["VEL_LOOKAHEAD"]
        self.finished = False

        if len(self.path) < 2 or end_idx - start_idx < 2:
            self.finished = True

        if self.end_idx > len(self.path) - 2:
            self.end_idx = len(self.path) - 2

    def get_action_go_to_position(self, target_pos, current_pos, current_yaw):

        target_pos_drone = pos_to_drone(current_pos, current_yaw, target_pos)
        target_heading = target_pos - current_pos
        target_yaw = vec_to_yaw(target_heading)

        diff_yaw = clip_angle(target_yaw - current_yaw)
        diff_x = target_pos_drone[0]
        diff_y = target_pos_drone[1]

        vel_x = diff_x * self.params["K_X"]
        vel_y = diff_y * self.params["K_Y"]
        vel_yaw = diff_yaw * self.params["K_Yaw"]

        action = np.asarray([vel_x, vel_y, vel_yaw])
        action = self.clip_vel(action)

        return action

    def get_action_go_to_position_cvel(self, target_pos, current_pos, current_yaw):
        # Get the action corresponding to going to a point further along the vector
        action = self.get_action_go_to_position(target_pos, current_pos, current_yaw)

        # Then just clip it to constant velocity
        if action[0] > self.params["Line_Epsilon"]:
            action[0] = self.dynamics_params["MAX_VEL_X"]
        elif action[0] < -self.params["Line_Epsilon"]:
            action[0] = -self.dynamics_params["MAX_VEL_X"]
        else:
            action[0] = 0.0

        action[1] = 0

        action = self.clip_vel(action)

        return action

    def get_action_follow_vector(self, start_pos, vector, current_pos, current_yaw):
        # Get the action corresponding to going to a point further along the vector
        action = self.get_action_go_to_position(current_pos + vector, current_pos, current_yaw)

        # Then just clip it to constant velocity
        if action[0] > self.params["Line_Epsilon"]:
            action[0] = self.params["Line_Vel_X"]
        elif action[0] < -self.params["Line_Epsilon"]:
            action[0] = -self.params["Line_Vel_X"]
        else:
            action[0] = 0.0

        if action[1] > self.params["Line_Epsilon"]:
            action[1] = self.params["Line_Vel_Y"]
        elif action[1] < -self.params["Line_Epsilon"]:
            action[1] = -self.params["Line_Vel_Y"]
        else:
            action[1] = 0.0

        # Also encourage the drone cancel any lateral error
        actual_direction = current_pos - start_pos
        actual_direction[2] = 0

        # TODO: Actual direction is often zero, producing a NaN!!
        dir_mag = np.linalg.norm(actual_direction)
        act_mag = np.linalg.norm(vector)

        # If we haven't moved at all yet.
        if (1e-9 > dir_mag > -1e-9) or (1e-9 > act_mag > -1e-9):
            return action

        v1 = vector / act_mag
        v2 = actual_direction / dir_mag

        cross = np.cross(v1, v2)

        # If we have veered off path, steer back on the path
        action[2] -= self.params["K_Yaw_Offset"] * cross[2]

        return action

    def reduce_speed_for_bendy_paths(self, action, current_pos, current_yaw, lookahead_pos):

        lookahead_dir = lookahead_pos - current_pos
        lookahead_dir = lookahead_dir / np.linalg.norm(lookahead_dir)
        lookahead_yaw = vec_to_yaw(lookahead_dir)

        diff = math.fabs(current_yaw - lookahead_yaw)
        diff = clip_angle(diff)
        diff = math.fabs(diff)
        diff /= 3.14159
        diff *= self.params["K_X_Lookahead_Reduction"]

        multiplier = 1 - diff
        if multiplier < 0.3:
            multiplier = 0.3
        if multiplier > 1:
            multiplier = 1.0
        #print("diff:", diff, "m:", multiplier)
        action[0] *= multiplier
        return action

    def reduce_speed_for_initial_acceleration(self, action):
        factor = float((self.current_step + 1) / self.params["ACCELERATE_STEPS"])
        if factor > 1.0:
            factor = 1.0
        action[0] *= factor
        return action

    def set_discrete(self, discrete):
        self.discrete = discrete

    def discretize_action(self, action):
        if action[0] > self.params["DISCRETE_EPSILON_X"]:
            action[0] = self.dynamics_params["MAX_VEL_X"]
        elif action[0] < -SETTINGS["DISCRETE_EPSILON_X"]:
            action[0] = -self.dynamics_params["MAX_VEL_X"]
        else:
            action[0] = 0
        return action

    def pos_from_state(self, state):
        return state[0:3]

    def vel_from_state(self, state):
        return state[6:9]

    def yaw_from_state(self, state):
        return state[5]


    def get_follow_path_action(self, state):
        if self.finished:
            print("Oracle: Already finished")
            return np.asarray([0, 0, 0, 1])

        start_pos = np.zeros(3)
        next_pos = np.zeros(3)
        lookahead_pos = np.zeros(3)
        vel_lookahead_pos = np.zeros(3)
        curr_pos = np.zeros(3)
        curr_vel = np.zeros(3)

        curr_pos[0:2] = self.pos_from_state(state)[0:2]
        curr_vel[0:2] = self.vel_from_state(state)[0:2]

        # Change the lookahead depending on drone's current velocity.
        percent_speed = np.linalg.norm(curr_vel) / self.dynamics_params["MAX_VEL_X"]
        self.current_lookahead = int(self.params["MIN_LOOKAHEAD"] + self.params["LOOKAHEAD"] * percent_speed)
        self.current_vel_lookahed = int(self.params["MIN_LOOKAHEAD"] + self.params["VEL_LOOKAHEAD"] * percent_speed)

        curr_yaw = self.yaw_from_state(state)

        self.control_step += 1

        action = np.zeros(3)
        start_pos[0:2] = self.path[self.current_step]

        # Advance the step counter along the path
        while True:
            # If there are no longer LOOKAHEAD steps available in the path,
            # keep reducing lookahead until we finish the path
            max_lookahead = max(len(self.path) - 2, self.end_idx) - self.current_step

            # Keep reducing lookahead until MIN_LOOKAHEAD when ending a segment
            if self.current_lookahead > self.params["MIN_LOOKAHEAD"]:
                max_lookahead = self.end_idx - self.current_step
            # If ending the entire path, reduce lookahead all the way to 0
            else:
                max_lookahead = max(len(self.path) - 2 - self.current_step, 0)

            if self.current_lookahead > max_lookahead:
                self.current_lookahead = max_lookahead
            if self.current_vel_lookahed > max_lookahead:
                self.current_vel_lookahed = max_lookahead

            next_pos[0:2] = self.path[self.current_step + 1]
            lookahead_pos[0:2] = self.path[self.current_step + self.current_lookahead]
            vel_lookahead_pos[0:2] = self.path[self.current_step + self.current_vel_lookahed]

            curr_direction = lookahead_pos - start_pos
            path_direction = lookahead_pos - start_pos
            threshold_dot = np.dot(next_pos, path_direction)

            # Must advance along path
            if np.dot(curr_pos, path_direction) > threshold_dot:
                self.current_step += 1

                #print ("Step = " + str(self.current_step) + "/" + str(self.end_idx))
                # If this is the end of path, mark as finished and return the final action
                if self.current_step >= self.end_idx:
                    self.current_lookahead = self.params["LOOKAHEAD"]
                    self.current_vel_lookahed = self.params["VEL_LOOKAHEAD"]
                    self.finished = True
                    print("Oracle: End of path reached after " + str(self.control_step) + " actions")
                    return np.asarray([0, 0, 0, 1])
            # We're good, let's compute the action
            else:
                break

        action = self.get_action_follow_vector(start_pos, curr_direction, curr_pos, curr_yaw)

        if self.discrete:
            action = self.discretize_action(action)

        action = DC.normalize_action(action)

        action = self.reduce_speed_for_bendy_paths(action, curr_pos, curr_yaw, vel_lookahead_pos)
        action = self.reduce_speed_for_bendy_paths(action, curr_pos, curr_yaw, lookahead_pos)
        action = self.reduce_speed_for_initial_acceleration(action)

        # Sometimes the speed reductions above cause the drone to stop completely. Don't let that happen:
        if action[0] < self.params["Min_Vel_X"]:# and action[0] > 0:
            action[0] = self.params["Min_Vel_X"]

        if np.linalg.norm(next_pos - curr_pos) > self.max_deviation and \
            np.linalg.norm(start_pos - curr_pos) > self.max_deviation and \
            np.linalg.norm(lookahead_pos - curr_pos) > self.max_deviation:
            return None

        #print ("Action: ", action)
        if np.isnan(action).any():
            print("ERROR! Oracle produced a NaN action!")
            return None

        print("Fancy Oracle Action : ", action)

        return np.asarray(list(action) + [0])

    # -----------------------------------------------------------------------------------------------------------------

    def start_sequence(self):
        pass

    def get_action(self, state, instruction=None):
        return self.get_follow_path_action(state.state)


