from policies.abstract_policy import AbstractPolicy
import numpy as np

from parameters.parameter_server import get_current_parameters


class BaselineAverage(AbstractPolicy):

    def __init__(self):
        params = get_current_parameters()["BaselineAverage"]
        self.current_step = 0
        self.avg_fwd_vel = params["AvgSpeed"]
        self.avg_yawrate = params["AvgYawrate"]
        self.avg_num_steps = params["AvgSteps"]

    def start_sequence(self):
        self.current_step = 0

    def get_action(self, state, instruction=None, sample=False, rl_rollout=False):
        action = np.zeros(4)
        action[0] = self.avg_fwd_vel
        action[2] = self.avg_yawrate

        # Stop after average number of steps
        self.current_step += 1
        if self.current_step >= self.avg_num_steps:
            action[3] = 1.0

        return action, {}
