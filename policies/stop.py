from policies.abstract_policy import AbstractPolicy
import numpy as np


class BaselineStop(AbstractPolicy):

    def __init__(self):
        pass

    def start_sequence(self):
        pass

    def get_action(self, state, instruction=None):
        action = np.zeros(4)
        action[3] = 1.0

        return action