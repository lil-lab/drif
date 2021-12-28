import torch
from torch import nn as nn
from tensorboardX import SummaryWriter

from drones.train_real.adversarial_networks import Discriminator, Critic
from drones.train_real.PerceptionModel import PerceptionModel
from learning.datasets.dataset_real_sim import RealSimDataset, ConcatRealSimDataset

from learning.modules.auxiliaries.class_auxiliary_2d import ClassAuxiliary2D
from drones.train_real.DomainAuxiliary import AdversaryAuxiliary

import parameters.parameter_server as P


class DomainWrapperModel(torch.nn.Module):
    # Used if perception model is only 1 network instead of 1 for simulated + 1 for real images.
    def __init__(self, run_name, is_cuda):
        super(DomainWrapperModel, self).__init__()
        self.model_name = "Wrapper"
        self.model_params = P.get_current_parameters()["Model"]
        self.aux_weights = P.get_current_parameters()["AuxWeightsPerception"]

        self.discriminator = Discriminator(run_name, grad_reversal=True)
        self.model = PerceptionModel(run_name, real=True)
        self.real_sim = True
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.train_writer = SummaryWriter(log_dir="runs/" + run_name + "/train")
        self.test_writer = SummaryWriter(log_dir="runs/" + run_name + "/test")

    def get_dataset(self, dataset_name, dataset_prefix, envs, eval):
        real_dataset = self.model_real.get_dataset(dataset_names=dataset_name, dataset_prefix=dataset_prefix, envs=envs, eval=eval, real=True)
        sim_dataset = self.model_sim.get_dataset(dataset_names=dataset_name, dataset_prefix=dataset_prefix, envs=envs, eval=eval, real=False)
        dataset = ConcatRealSimDataset(real_dataset, sim_dataset)
        return dataset

    def cuda(self, device=None):
        torch.nn.Module.cuda(self, device)
        self.model_real.cuda(device)
        self.model_sim.cuda(device)
        self.discriminator.cuda(device)
        return self

    def init_weights(self):
        self.model_real.init_weights()
        self.model_sim.init_weights()

    def get_iter(self):
        return int(self.iter.data.item())

    def inc_iter(self):
        self.iter += 1
        self.model_sim.inc_iter()
        self.model_real.inc_iter()
        self.discriminator.inc_iter()

    def sup_loss_on_batch(self, batch):
        batch_real = batch

        images_real = batch_real[0]

        features_real = self.model_real(images_real)
        real_metrics = self.model_real.sup_loss_on_batch(batch_real, features=features_real)

        batch_metrics = {}
        for key in real_metrics.keys():
            batch_metrics[key] = real_metrics[key]

        return batch_metrics


class DomainWrapperModel2Nets(torch.nn.Module):
    def __init__(self, run_name, is_cuda, wasserstein=False):
        super(DomainWrapperModel2Nets, self).__init__()
        self.model_name = "Wrapper2"
        self.model_params = P.get_current_parameters()["Model"]
        self.aux_weights = P.get_current_parameters()["AuxWeightsPerception"]

        self.model_real = PerceptionModel(run_name, real=True, domain_loss_type="adversarial")
        self.model_sim = PerceptionModel(run_name, real=False, domain_loss_type="adversarial")
        self.real_sim = True
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.wasserstein = wasserstein
        if self.wasserstein:
            # Using Wasserstein GAN
            self.discriminator = Critic(run_name, grad_reversal=True)

        else:
            # Not using Wasserstein GAN
            self.discriminator = Discriminator(run_name, grad_reversal=True)

        #TODO: implement classif_dropout this in the parameters
        classif_dropout = 0.5
        # 1 classifier is used to classify both real and simulated images
        self.one_classifier = ClassAuxiliary2D("aux_class",
                                               None,
                                               self.model_params["feature_channels"],
                                               self.model_params["num_landmarks"],
                                               classif_dropout,
                                               "fpv_features", "lm_pos_fpv", "lm_indices")

        # Auxiliary domain loss associated to the critic/discriminator
        self.domain_auxiliary = AdversaryAuxiliary(name="aux_domain", adversarial_network=self.discriminator)

        self.train_writer = SummaryWriter(log_dir="runs/" + run_name + "/train")
        self.test_writer = SummaryWriter(log_dir="runs/" + run_name + "/test")

    def get_dataset(self, dataset_name, envs, eval):
        # dataset contains both real and simulated images. Images don't have to be aligned when using discriminative loss
        real_dataset = self.model_real.get_dataset(dataset_name=dataset_name, envs=envs, eval=eval, real=True)
        sim_dataset = self.model_sim.get_dataset(dataset_name=dataset_name, envs=envs, eval=eval, real=False)
        realsim_dataset = RealSimDataset(real_dataset, sim_dataset)
        return realsim_dataset

    def cuda(self, device=None):
        torch.nn.Module.cuda(self, device)
        self.model_real.cuda(device)
        self.model_sim.cuda(device)
        self.discriminator.cuda(device)
        self.one_classifier.cuda(device)
        return self

    def init_weights(self):
        self.model_real.init_weights()
        self.model_sim.init_weights()

    def get_iter(self):
        return int(self.iter.data.item())

    def inc_iter(self):
        self.iter += 1
        self.model_sim.inc_iter()
        self.model_real.inc_iter()
        self.discriminator.inc_iter()

    def get_lm_indices_and_pos(self, batch):
        '''
        Extract indices and landmarks and converts the landmarks' positions on the image to positions on the feature map.
        :param batch: 1 batch of dataset
        :return: lm_indices_list: list of indices
        lm_pos_list: list of landmark positions.
        '''
        lm_pos_list, lm_indices_list = batch["lm_pos_fpv"], batch["lm_indices"]
        lm_indices_list = [self.model_real.cuda_var(x) for x in lm_indices_list]
        lm_pos_list = [self.model_real.cuda_var((x / 8).long()) for x in lm_pos_list]
        return lm_indices_list, lm_pos_list

    def sup_loss_on_batch(self, batch):
        # Add label: for further use. 0 for simulated, 1 for real.
        batch_real = batch["real"]
        batch_real["labels"] = [1. for _ in batch["real"]["images"]]

        batch_sim = batch["sim"]
        batch_sim["labels"] =[0. for _ in batch["sim"]["images"]]

        images_real = batch_real["images"]
        images_sim = batch_sim["images"]

        features_real = self.model_real(images_real)
        features_sim = self.model_sim(images_sim)

        if self.wasserstein:
            # Wasserstein critic does not return accuracy
            domain_loss = self.domain_auxiliary(features_real, features_sim)
        else:
            domain_loss, domain_accuracy = self.domain_auxiliary(features_real, features_sim)

        lm_indices_list_real, lm_pos_list_real = self.get_lm_indices_and_pos(batch_real)
        lm_indices_list_sim, lm_pos_list_sim = self.get_lm_indices_and_pos(batch_sim)

        real_class_loss, real_class_accuracy, _ = self.one_classifier(features_real, lm_pos_list_real,
                                                                   lm_indices_list_real)
        sim_class_loss, sim_class_accuracy, _ = self.one_classifier(features_sim, lm_pos_list_sim, lm_indices_list_sim)

        real_metrics = self.model_real.sup_loss_on_batch(batch_real, features=features_real.detach())
        sim_metrics = self.model_sim.sup_loss_on_batch(batch_sim, features=features_sim.detach())

        # Compute "crossed" metrics ie real features on classifier of the sim features and inversely.
        # Those metrics are not used for training but just to know how the training goes.
        cross_metrics = self.cross_evaluate_domain(features_real.detach(), features_sim.detach(), lm_pos_list_real, lm_pos_list_sim,
                                                   lm_indices_list_real, lm_indices_list_sim)

        # Backward on this loss just trains the classifier
        two_class_losses = real_metrics["/loss/class"] + sim_metrics["/loss/class"]

        one_class_loss = real_class_loss + sim_class_loss
        one_class_accuracy = (sim_class_accuracy + real_class_accuracy) / 2

        # Save metrics in dictionary batch_metrics

        domain_weight_key = "aux_domain_wasserstein" if self.wasserstein else "aux_domain_adversarial"
        batch_metrics = {}
        batch_metrics["/shared/total_loss"] = one_class_loss + domain_loss * self.aux_weights[domain_weight_key] + \
                                             two_class_losses
        batch_metrics["/shared/domain_loss"] = domain_loss

        if not self.wasserstein:
            batch_metrics["/shared/domain_accuracy"] = domain_accuracy

        batch_metrics["/one_classifier/loss"] = one_class_loss
        batch_metrics["/one_classifier/accuracy"] = one_class_accuracy

        # The following metrics are metrics not used for training (incl. classifiers loss and accuracies for train or sim data)
        for key in real_metrics.keys():
            batch_metrics["/real_model" + key] = real_metrics[key]
        for key in sim_metrics.keys():
            batch_metrics["/sim_model" + key] = sim_metrics[key]
        for key in cross_metrics.keys():
            batch_metrics[key] = cross_metrics[key]
        return batch_metrics

    def cross_evaluate_domain(self, features_real, features_sim, lm_pos_list_real, lm_pos_list_sim,
                              lm_indices_list_real, lm_indices_list_sim):
        sim_loss_on_real, sim_accuracy_on_real, _ = self.model_sim.class_auxiliary(features_real, lm_pos_list_real,
                                                                     lm_indices_list_real)
        real_loss_on_sim, real_accuracy_on_sim, _ = self.model_real.class_auxiliary(features_sim, lm_pos_list_sim,
                                                                             lm_indices_list_sim)

        evaluate_metrics = {"/sim_model/loss/real_data": sim_loss_on_real, "/real_model/loss/sim_data": real_loss_on_sim,
                            "/sim_model/accuracy/real_data": sim_accuracy_on_real, "/real_model/accuracy/sim_data": real_accuracy_on_sim}
        return evaluate_metrics

