from policies.abstract_policy import AbstractPolicy
import numpy as np

#TODO: Possibly delete this

class ModelPolicy(AbstractPolicy):

    def __init__(self, model):
        self.model = model

    def start_sequence(self):
        self.model.start_sequence()

    def get_action(self, state, instruction=None):
        action = np.zeros(3)
        action[0] = 0.83
        return action, False