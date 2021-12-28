import numpy as np
import math

from data_io.env import load_and_convert_path
from data_io.instructions import get_instruction_segment

from geometry import vec_to_yaw, clip_angle, yaw_to_vec
from policies.abstract_policy import AbstractPolicy
from parameters.parameter_server import get_current_parameters
from data_io.instructions import debug_untokenize_instruction


class BasicCarrotPlanner(AbstractPolicy):
    """
    Very simple local planner that applies a P-control rule to fly towards a point a fixed distance ahead on the path
    """
    def __init__(self, path=None, max_deviation=500):

        self.path = path
        self.end_idx = 0
        self.params = get_current_parameters()["BasicCarrotPlanner"]

        self.current_step = 0
        self.control_step = 0

        self.finished = False
        self.max_deviation = max_deviation
        self.last_sanity_pos = np.zeros(3)
        self.set_path(path)

        self.ros = get_current_parameters()["Setup"]["real_drone"]
        self.rviz = None
        if self.ros:
            from drones.aero_interface.rviz import RvizInterface
            self.rviz = RvizInterface(
                base_name="/bcp/",
                path_topics=["ground_truth"],
                markerarray_topics=["instruction"])

    def set_path(self, path):
        self.path = path
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
        self.end_idx = end_idx
        self.current_step = start_idx
        self.control_step = 0
        self.finished = False
        if self.ros:
            self.rviz.create_path_from_2d_array("ground_truth", self.path[start_idx:end_idx], publish=True)

        if len(self.path) < 2 or end_idx - start_idx < 2:
            self.finished = True

        if self.end_idx > len(self.path) - 2:
            self.end_idx = len(self.path) - 2

    def start_rollout(self, env_id, set_idx, seg_idx):
        path = load_and_convert_path(env_id)
        seg = get_instruction_segment(env_id, set_idx, seg_idx)
        self.set_path(path)
        self.set_current_segment(seg["start_idx"], seg["end_idx"])

    def get_action_go_to_carrot(self, curr_pos, curr_yaw, carrot_pos):
        #print(f"BCP: curr_pos: {curr_pos}  curr_yaw: {curr_yaw}  carrot_pos: {carrot_pos}")
        carrot_dir = carrot_pos - curr_pos
        carrot_yaw = vec_to_yaw(carrot_dir)
        delta_yaw = carrot_yaw - curr_yaw
        delta_yaw = clip_angle(delta_yaw)

        #print(f"BCP: Carrot pos: {carrot_pos}, current pos: {curr_pos}")
        #print(f"BCP: Carrot yaw: {carrot_yaw}, current yaw: {curr_yaw}")

        vel_x = self.params["vel_x"] / (1 + 2 * math.fabs(delta_yaw) / math.pi)
        #vel_x = self.params["VEL_X"]
        vel_yaw = self.params["k_yaw"] * delta_yaw

        action = np.asarray([vel_x, 0, vel_yaw])
        return action

    def reduce_speed_for_initial_acceleration(self, action):
        factor = float((self.current_step + 1) / self.params["accelerate_steps"])
        if factor > 1.0:
            factor = 1.0
        action[0] *= factor
        return action

    def reduce_speed_for_path_end(self, action, dst_to_end):
        target_dst_end = self.params["end_dst"]
        if dst_to_end < target_dst_end:
            action[0] = action[0] * (dst_to_end / target_dst_end)
        return action

    def reduce_speed_for_turns(self, action, curr_pos, curr_yaw, carrot_pos):
        carrot_yaw = vec_to_yaw(carrot_pos - curr_pos)
        yaw_diff = clip_angle(carrot_yaw - curr_yaw)
        scaling = math.exp(-math.fabs(yaw_diff))
        action[0] = action[0] * scaling
        return action

    def pos_from_state(self, state):
        return state[0:3]

    def vel_from_state(self, state):
        return state[6:9]

    def yaw_from_state(self, state):
        return state[5]

    def find_carrot_pos(self, curr_pos):
        prev_carrot_pos = self.path[self.current_step]
        carrot_pos = prev_carrot_pos
        carrot_idx = self.current_step
        cumdist = 0
        for i in range(self.current_step, self.end_idx):
            carrot_pos = self.path[i]
            carrot_idx = i
            dst_to_prev = np.linalg.norm(carrot_pos - prev_carrot_pos)
            cumdist += dst_to_prev
            #print("dst: ", dst, curr_pos, carrot_pos)
            if cumdist > self.params["lookahead_dst"]:
                break
            prev_carrot_pos = carrot_pos
        #print("carrot idx: ", self.current_step, carrot_idx, cumdist, self.params["LOOKAHEAD_DST"])
        return carrot_pos, carrot_idx

    def exceeded_max_deviation(self, curr_pos, dst_to_carrot):
        current_path_pt = self.path[self.current_step]
        dst = np.linalg.norm(curr_pos - current_path_pt)
        return dst > self.max_deviation and dst_to_carrot > self.max_deviation

    def get_follow_path_action(self, state):
        if self.finished:
            #print("BCP: Already finished")
            return np.asarray([0, 0, 0, 1])

        start_pos = np.zeros(3)
        curr_pos = np.zeros(2)
        curr_pos[0:2] = self.pos_from_state(state)[0:2]
        curr_yaw = self.yaw_from_state(state)
        carrot_pos, carrot_idx = self.find_carrot_pos(curr_pos)
        dst_to_carrot = np.linalg.norm(carrot_pos - curr_pos)
        dst_to_end = np.linalg.norm(self.path[self.end_idx] - curr_pos)

        self.control_step += 1

        start_pos[0:2] = self.path[self.current_step]

        # Advance along the path
        end = False
        while True:
            this_step = self.path[self.current_step]
            next_idx = self.current_step + 1
            hasnext = next_idx <= self.end_idx
            if hasnext:
                next_step = self.path[next_idx]
                dst_to_next = np.linalg.norm(next_step - curr_pos)
            else:
                end = True
                break

            dst_to_this = np.linalg.norm(this_step - curr_pos)
            if dst_to_next <= dst_to_this + 1e-3:
                self.current_step += 1
                #print(f"BCP Next {self.current_step}")
            elif dst_to_carrot <= dst_to_this + 1e-3:
                self.current_step = carrot_idx
                break
            else:
                break   # Remain in the current step

        # Check if end of path reached:
        if end or dst_to_end < self.params["stop_dst"]:
            #print(f"BCP: pos: {curr_pos}  next point: {next_step}  dst: {dst_to_end}  end: {end}")
            self.finished = True
            print("BCP: End of path reached after " + str(self.control_step) + " actions")
            return np.asarray([0, 0, 0, 1])

        action = self.get_action_go_to_carrot(curr_pos, curr_yaw, carrot_pos)
        action = self.reduce_speed_for_initial_acceleration(action)
        action = self.reduce_speed_for_path_end(action, dst_to_end)
        if self.params["decelerate_for_turns"]:
            action = self.reduce_speed_for_turns(action, curr_pos, curr_yaw, carrot_pos)
        action = self.clip_vel(action)

        # Sometimes the speed reductions above cause the drone to stop completely. Don't let that happen:
        if action[0] < self.params["min_vel_x"]:# and action[0] > 0:
            action[0] = self.params["min_vel_x"]

        if self.exceeded_max_deviation(curr_pos, carrot_idx):
            print("Exceeded max deviation of: ", self.max_deviation)
            return None

        if np.isnan(action).any():
            print("ERROR! Oracle produced a NaN action!")
            return None

        return np.asarray(list(action) + [0])

    # -----------------------------------------------------------------------------------------------------------------

    def start_sequence(self):
        pass

    def get_action(self, state, instruction=None, sample=False, rl_rollout=False):
        if self.rviz is not None:
            self.rviz.publish_instruction_text("instruction", instruction if isinstance(instruction, str) else debug_untokenize_instruction(instruction))
        return self.get_follow_path_action(state.state), {}


