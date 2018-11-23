from policies.abstract_policy import AbstractPolicy
import numpy as np

from parameters.parameter_server import get_current_parameters


class BaselineStraight(AbstractPolicy):

    def __init__(self):
        params = get_current_parameters()["BaselineStraight"]
        self.current_step = 0
        self.avg_fwd_vel = params["AvgSpeed"]
        self.avg_num_steps = params["AvgSteps"]

    def start_sequence(self):
        self.current_step = 0

    def get_action(self, state, instruction=None):
        action = np.zeros(4)
        action[0] = self.avg_fwd_vel

        # Stop after average number of steps
        self.current_step += 1
        if self.current_step >= self.avg_num_steps:
            action[3] = 1.0

        return action