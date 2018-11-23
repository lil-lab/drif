import torch
from torch import nn as nn
from torch.autograd import Variable

from utils.dict_tools import dict_cross_map

from learning.modules.auxiliary_objective_base import AuxiliaryObjective
from learning.modules.cuda_module import CudaModule


class ModuleWithAuxiliaries(CudaModule):
    def __init__(self):
        super(ModuleWithAuxiliaries, self).__init__()
        self.aux_keys = []
        self.auxiliaries = {}
        self.inputs = {}

    def reset(self):
        self.inputs = {}

    def cuda(self, device=None):
        CudaModule.cuda(self, device)
        for key, aux in self.auxiliaries.items():
            aux.cuda(device)

    def input_required(self, input_name):
        for aux_key, aux in self.auxiliaries.items():
            if input_name in aux.get_required_inputs():
                return True
        return False

    def add_auxiliary(self, auxiliary_objective, key=None):
        """
        Adds an auxiliary objective, which is a subclass of auxiliary_objective_base
        :param auxiliary_objective:
        :param key:
        :return:
        """
        if key is None:
            key = auxiliary_objective.get_name()
        self.auxiliaries[key] = auxiliary_objective
        self.add_module(key, auxiliary_objective)
        self.aux_keys.append(key)

    def keep_input(self, key, input):
        """
        Stores a tensor for later retrieval with a given key
        :param key:
        :param input:
        :return:
        """
        if key not in self.inputs:
            self.inputs[key] = []
        self.inputs[key].append(input)

    def keep_inputs(self, key, input):
        """
        Stores a batch or sequence of tensors for later retrieval with a given key
        :param key:
        :param input:
        :return:
        """
        if type(input) == Variable or type(input) == torch.Tensor:
            for i in range(input.size(0)):
                self.keep_input(key, input[i:i+1])
        elif type(input) == list:
            for i in range(len(input)):
                inp = input[i]
                if type(inp) is Variable:
                    inp = inp.unsqueeze(0)
                self.keep_input(key, inp)
        else:
            raise Exception("ModuleWithAuxiliaries: Unrecognized input: " + str(type(input)))

    def get_latest_input(self, key):
        """
        Retrieves a the latest previously stored tensor with the given key
        :param key:
        :return:
        """
        if key in self.inputs:
            return self.inputs[key][-1]
        return None

    def get_inputs_batch(self, key, cat=False):
        """
        Retrieves all tensors with the given key, stacked in batch
        :param key:
        :return:
        """
        if key not in self.inputs:
            return None
        if cat:
            return torch.cat(self.inputs[key], dim=0)
        else:
            return torch.stack(self.inputs[key], dim=0)

    def clear_inputs(self, key):
        """
        Removes all stored tensors associated with the given key
        :param key:
        :return:
        """
        if key in self.inputs:
            del self.inputs[key]

    def print_auxiliary_info(self):
        print("Using auxiliary objectives:")
        for key in self.auxiliaries:
            print("       - " + key)

    def calculate_aux_loss(self, reduce_average=False):
        """
        Evaluates all auxiliary objectives, taking their inputs from the kept inputs (from keep_input calls)
        Returns their losses in a dictionary
        :param targets: Dict, where keys are auxiliary names and values are lists of labels.
            For each auxiliary, the number of labels provided must match the total number of inputs previously stored
            If a given auxiliary doesn't require a target value, then it's key can be omitted
        :return: Dict, where keys are auxiliary names and values are Variables with the total loss value
        """
        loss_dict = {}
        count_dict = {}
        for module_name in self.aux_keys:
            input_names = self.auxiliaries[module_name].get_required_inputs()
            input_list = []
            for input_name in input_names:
                input_list.append(self.inputs[input_name])
            #input_list = list(zip(*input_list))
            # Input list is a list of lists, where outer list is over timesteps and inner list is over inputs to the auxiliary

            ret_vals = self.auxiliaries[module_name](*input_list)
            if len(ret_vals) == 2:
                loss, count = ret_vals
            elif len(ret_vals) == 3:
                loss, _, count = ret_vals
            if loss is None:
                continue

            if module_name in loss_dict:
                loss_dict[module_name] += loss
                count_dict[module_name] += count
            else:
                loss_dict[module_name] = loss
                count_dict[module_name] = count

            """
            # TODO: Automatically batch this
            for i in range(len(input_list)):
                inputs = input_list[i]
                if any([e is None for e in inputs]):
                    continue

                #print("Computing Auxiliary: ", module_name)
                loss = self.auxiliaries[module_name](*inputs)
                if module_name in loss_dict:
                    loss_dict[module_name] += loss
                    count_dict[module_name] += 1
                else:
                    loss_dict[module_name] = loss
                    count_dict[module_name] = 1
            """


        if reduce_average:
            avg_loss_dict = dict_cross_map(loss_dict, count_dict, lambda a, b: a / (b + 1e-9))
            return avg_loss_dict
        else:
            return loss_dict, count_dict

    def combine_aux_losses(self, aux_losses, loss_weights):
        """
        Takes a dictionary of auxiliary losses and a dictionary of associated weights, where weights and losses
        are identified by the keys of the auxiliary objectives from which they came from.
        Outputs a single loss value, which is a convex combination of auxiliary losses with the given weights
        :param aux_losses:
        :param loss_weights:
        :return:
        """
        total_loss = None
        for key in aux_losses:
            weight = 1
            if key in loss_weights:
                weight = loss_weights[key]
            else:
                raise Exception("Auxiliary weight not defined for " + str(key))
            this_loss = aux_losses[key] * weight
            if total_loss is None:
                total_loss = this_loss
            else:
                total_loss += this_loss
        if total_loss is None:
            return 0
        return total_loss
