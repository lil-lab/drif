import math

import numpy as np

import pomdp.convert_action as DC
from geometry import vec_to_yaw, clip_angle, pos_to_drone
from policies.abstract_policy import AbstractPolicy
from parameters.parameter_server import get_current_parameters

from utils.paths import condense_path_with_mapping


class SimpleCarrotPlanner(AbstractPolicy):

    def __init__(self, path=None, max_deviation=500):
        if path is not None:
            self.path, self.path_idx_map = condense_path_with_mapping(path)
        else:
            self.path = None
            self.path_idx_map = None
        self.end_idx = 0
        self.params = get_current_parameters()["SimpleCarrotPlanner"]

        self.current_step = 0
        self.control_step = 0

        self.current_lookahead = self.params["lookahead"]
        self.finished = False
        self.max_deviation = max_deviation

        self.last_sanity_pos = np.zeros(3)
        self.set_path(path)

    def set_path(self, path):
        if path is None:
            return
        condensed_path, path_map = condense_path_with_mapping(path)
        self.path = condensed_path
        self.path_idx_map = path_map
        if path is not None:
            self.end_idx = len(path) - 2

    def clip_vel(self, action):
        # Never go backwards
        if action[0] > 1:
            action[0] = 1
        if action[0] < 0:
            action[0] = 0

        if action[2] > 1:
            action[2] = 1
        if action[2] < -1:
            action[2] = -1
        return action

    def set_current_segment(self, start_idx, end_idx):
        self.end_idx = self.path_idx_map[end_idx]
        self.current_step = self.path_idx_map[start_idx]
        self.current_lookahead = self.params["lookahead"]
        self.finished = False

        if len(self.path) < 2 or end_idx - start_idx < 2:
            self.finished = True

        if self.end_idx > len(self.path) - 2:
            self.end_idx = len(self.path) - 2

    def get_action_go_to_position(self, target_pos, current_pos, current_yaw):

        target_pos_drone = pos_to_drone(current_pos, current_yaw, target_pos)
        target_heading = target_pos - current_pos
        target_yaw = vec_to_yaw(target_heading)

        diff_yaw_raw = target_yaw - current_yaw
        diff_yaw = clip_angle(diff_yaw_raw)
        diff_x = target_pos_drone[0]

        vel_x = self.params["vel_x"]
        vel_yaw = diff_yaw * self.params["k_yaw"]

        action = np.asarray([vel_x, 0, vel_yaw])
        action = self.clip_vel(action)

        return action

    def get_action_follow_vector(self, start_pos, vector, current_pos, current_yaw):
        # Get the action corresponding to going to a point further along the vector
        action = self.get_action_go_to_position(current_pos + vector, current_pos, current_yaw)

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
        action[2] -= self.params["k_offset"] * cross[2]

        return action

    def reduce_speed_for_initial_acceleration(self, action):
        factor = float((self.current_step + 1) / self.params["accelerate_steps"])
        if factor > 1.0:
            factor = 1.0
        action[0] *= factor
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
        curr_pos = np.zeros(3)

        curr_pos[0:2] = self.pos_from_state(state)[0:2]
        curr_yaw = self.yaw_from_state(state)

        # Change the lookahead depending on drone's current velocity.
        self.current_lookahead = self.params["lookahead"]

        self.control_step += 1

        action = np.zeros(3)
        start_pos[0:2] = self.path[self.current_step]

        # Advance the step counter along the path
        while True:
            # If there are no longer LOOKAHEAD steps available in the path,
            # keep reducing lookahead until we finish the path
            max_lookahead = min(len(self.path) - 2, self.end_idx) - self.current_step
            if self.current_lookahead > max_lookahead:
                self.current_lookahead = max_lookahead

            next_pos[0:2] = self.path[self.current_step + 1]
            try:
                lookahead_pos[0:2] = self.path[self.current_step + self.current_lookahead]
            except Exception as e:
                print("Error looking ahead")

            curr_direction = lookahead_pos - curr_pos
            path_direction = lookahead_pos - start_pos
            threshold_dot = np.dot(next_pos, path_direction)

            # Must advance along path
            if np.dot(curr_pos, path_direction) > threshold_dot:
                self.current_step += 1

                #print ("Step = " + str(self.current_step) + "/" + str(self.end_idx))
                # If this is the end of path, mark as finished and return the final action
                if self.current_step >= self.end_idx:
                    self.current_lookahead = self.params["lookahead"]
                    self.finished = True
                    print("Oracle: End of path reached after " + str(self.control_step) + " actions")
                    return np.asarray([0, 0, 0, 1])
            # We're good, let's compute the action
            else:
                break

        # TODO: What happens if we set curr_direction in terms of current position?
        action = self.get_action_follow_vector(start_pos, curr_direction, curr_pos, curr_yaw)
        action = self.reduce_speed_for_initial_acceleration(action)
        action = self.clip_vel(action)

        # Sometimes the speed reductions above cause the drone to stop completely. Don't let that happen:
        if action[0] < self.params["min_vel_x"]:# and action[0] > 0:
            action[0] = self.params["min_vel_x"]

        if np.linalg.norm(next_pos - curr_pos) > self.max_deviation and \
            np.linalg.norm(start_pos - curr_pos) > self.max_deviation and \
            np.linalg.norm(lookahead_pos - curr_pos) > self.max_deviation:
            print("Exceeded max deviation of: ", self.max_deviation)
            print(" curr pos: ", curr_pos, "next_pos: ", next_pos)
            return None

        #print ("Action: ", action)
        if np.isnan(action).any():
            print("ERROR! Oracle produced a NaN action!")
            return None

        #print("Oracle action : ", action)

        return np.asarray(list(action) + [0])

    # -----------------------------------------------------------------------------------------------------------------

    def start_sequence(self):
        pass

    def get_action(self, state, instruction=None):
        return self.get_follow_path_action(state.state)


