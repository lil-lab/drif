import numpy as np
import torch


class AbstractIncrementalModel(object):
    def __init__(self, config, constants):
        self.config = config
        self.constants = constants

    def load_saved_model(self, load_dir):
        raise NotImplementedError()

    def save_model(self, save_dir):
        raise NotImplementedError()

    def get_probs_batch(self, agent_observed_state_list, mode=None, volatile=False):
        """
        :param agent_observed_state_list: list of agent observed states
        :type agent_observed_state_list: list
        :return: PyTorch Variable of shape "BatchSize x NumActions"
        """
        raise NotImplementedError()

    def get_probs(self, agent_observed_state, model_state, mode=None, volatile=False):
        """
        :param agent_observed_state: single agent observed state
        :param model_state: model state containing previous values
        :param mode: mode for read pointer agent.
        :type agent_observed_state: AgentObservedState
        :return: PyTorch Variable of shape "NumActions"
        """
        raise NotImplementedError()
        # return self.get_probs_batch([agent_observed_state], mode=mode).view(-1)

    def eval_probs_batch(self, agent_observed_state_list, action_list,
                         mode=None):
        """
        :param agent_observed_state_list: list of agent observed states
        :type agent_observed_state_list: list
        :param action_list: list of actions taken (same length as AOS list),
            assumed to be integers in range(0, num_actions-1)
        :type action_list: list
        :return: numpy array of shape "BatchSize"
        """
        probs_batch = self.get_probs_batch(agent_observed_state_list,
                                           mode=mode).data
        action_batch = torch.from_numpy(np.array(action_list))
        chosen_action_probs = probs_batch.gather(1, action_batch.view(-1, 1))
        return chosen_action_probs.numpy()

    def eval_probs(self, agent_observed_state, action, mode=None):
        """
        :param agent_observed_state: single agent observed state
        :type agent_observed_state: AgentObservedState
        :param action: single action (assumed to be integer in
            range(0, num_actions-1)
        :type action: int
        :return: probability of taking action in given state (float)
        """
        action_prob_array = self.eval_probs_batch([agent_observed_state],
                                                  [action], mode=mode)
        return action_prob_array[0]

    def get_parameters(self):
        raise NotImplementedError()

    def share_memory(self):
        raise NotImplementedError()
