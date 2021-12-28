import sys

import torch
import torch.optim as optim
import numpy as np

from utils.simple_profiler import SimpleProfiler
from learning.utils import get_n_params, get_n_trainable_params
import parameters.parameter_server as P
# TODO: put get_stats in a proper file

class PerceptionTrainer:
    def __init__(
            self,
            model,
            state=None,
            epoch=0,
            name="",
            run_name="",
            #real_sim=False
    ):

        self.params = P.get_current_parameters()["GeneratorOptimizer"]
        self.batch_size = self.params['batch_size']
        self.weight_decay = self.params['weight_decay']
        self.optimizer = self.params['optimizer']
        self.num_loaders = self.params['num_loaders']
        self.lr = self.params['lr']
        self.betas = self.params['betas']
        self.name = name

        n_params = get_n_params(model)
        n_params_tr = get_n_trainable_params(model)
        print("Training Model:")
        print("Number of model parameters: " + str(n_params))
        print("Trainable model parameters: " + str(n_params_tr))

        self.model = model
        self.run_name = run_name
        # parameters of the discriminator are not included
        model_param = self.get_model_parameters(self.model.model_real) + \
                      self.get_model_parameters(self.model.model_sim) + \
                      self.get_model_parameters(self.model.one_classifier)

        if self.optimizer == "adam":
            self.optim = optim.Adam(model_param, self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            self.optim = optim.SGD(model_param, self.lr, weight_decay=self.weight_decay, momentum=0.9)

        self.train_epoch_num = epoch
        self.train_segment = 0
        self.test_epoch_num = epoch
        self.test_segment = 0
        self.set_state(state)
        self.batch_num = 0

    def get_model_parameters(self, model):
        params_out = []
        skipped_params = 0
        for param in model.parameters():
            if param.requires_grad:
                params_out.append(param)
            else:
                skipped_params += 1
        print(str(skipped_params) + " parameters frozen")
        return params_out

    def get_state(self):
        state = {}
        state["name"] = self.name
        state["train_epoch_num"] = self.train_epoch_num
        state["train_segment"] = self.train_segment
        state["test_epoch_num"] = self.test_epoch_num
        state["test_segment"] = self.test_segment
        return state

    def set_state(self, state):
        if state is None:
            return
        self.name = state["name"]
        self.train_epoch_num = state["train_epoch_num"]
        self.train_segment = state["train_segment"]
        self.test_epoch_num = state["test_epoch_num"]
        self.test_segment = state["test_segment"]

    def write_grad_summaries(self, writer, named_params, idx):
        for name, parameter in named_params:
            weights = parameter.data.cpu()
            mean_weight = torch.mean(weights)
            weights = weights.numpy()
            writer.add_histogram(self.model.model_name + "_internals" + "/hist_" + name + "_data", weights, idx, bins=100)
            writer.add_scalar(self.model.model_name + "_internals" + "/mean_" + name + "_data", mean_weight, idx)
            if parameter.grad is not None:
                grad = parameter.grad.data.cpu()
                mean_grad = torch.mean(grad)
                grad = grad.numpy()
                writer.add_histogram(self.model.model_name + "_internals" + "/hist_" + name + "_grad", grad, idx, bins=100)
                writer.add_scalar(self.model.model_name + "_internals" + "/mean_" + name + "_grad", mean_grad, idx)

    def write_grouped_loss_summaries(self, writer, losses, idx):
        pass

    def update_metrics(self, epoch_metrics, batch_metrics, i):
        assert(i >= 2)
        for key, value in zip(epoch_metrics.keys(), epoch_metrics.values()):
            epoch_metrics[key] = ((i-1) * value + batch_metrics[key].data.item())/i
        return epoch_metrics

    def update_count_dic(self, epoch_counts, batch_counts):
        for key in batch_counts.keys():
            if not (key in epoch_counts.keys()):
                epoch_counts[key] = batch_counts[key]
            else:
                epoch_counts[key] = batch_counts[key]
        return epoch_counts

    def train_epoch(self, dataloader, eval=False, critic_trainer=None):
        if eval:
            self.model.eval()
            inference_type = "eval"
            epoch_num = self.train_epoch_num
            self.test_epoch_num += 1
        else:
            self.model.train()
            inference_type = "train"
            epoch_num = self.train_epoch_num
            self.train_epoch_num += 1


        num_samples = len(dataloader.dataset)
        if num_samples == 0:
            print ("DATASET HAS NO DATA!")
            return -1.0

        num_batches = int((num_samples + self.batch_size - 1) / self.batch_size)

        PROFILE = False
        prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

        prof.tick("out")

        epoch_counts = {}


        i = 0
        data_iter = iter(dataloader)

        while i < len(dataloader):
            batch = data_iter.next()
            i0 = i
            prof.tick("batch_load")
            # Zero gradients before each segment and initialize zero segment loss
            self.optim.zero_grad()

            #batch_counts = get_stats_total(batch)
            #self.update_count_dic(epoch_counts, batch_counts)
            if True:
            #try:
                ######## Update critic (in case of Wasserstein critic) for a few batches ###########
                if (not eval) & self.model.wasserstein:
                    # In Wasserstein GAN, the training of the feature extractor is separated from the training of
                    # the critic (as opposed to non Wasserstein
                    i, critic_batch_metrics = critic_trainer.train_epoch(i)

                # Classification & domain losses
                batch_metrics = self.model.sup_loss_on_batch(batch)

                # metrics are stored and updated
                if i0 == 0:
                    epoch_metrics = dict(zip(batch_metrics.keys(), [x.data.item() for x in batch_metrics.values()]))
                else:
                    epoch_metrics = self.update_metrics(epoch_metrics, batch_metrics, i+1)

                batch_loss = batch_metrics["/shared/total_loss"]
                if type(batch_loss) == int:
                    print("Ding")

                prof.tick("forward")

                # Backprop and step
                if not eval:
                    batch_loss.backward()

                    prof.tick("backward")

                    # This is SLOW! Don't do it often
                    # TODO: Get rid of tensorboard
                    #if self.batch_num % 20 == 0 and hasattr(self.model, "writer"):
                    #    params = self.model.named_parameters()
                    #    self.write_grad_summaries(self.model.writer, params, self.batch_num)

                    self.batch_num += 1
                    self.optim.step()

                    prof.tick("optim")

                i += 1
                sys.stdout.write(
                    "\r Batch:" + str(i) + " / " + str(num_batches) + " loss: " + str(batch_loss.data.item()))
                sys.stdout.flush()

                self.train_segment += 0 if eval else 1
                self.test_segment += 1 if eval else 0

                prof.tick("rep")

                prof.loop()
                prof.print_stats(10)
            #except Exception as e:
            #    print("Exception encountered during batch update")
            #    print(e)

        #except Exception as e:
            #print("Error during epoch training")
            #print(e)
            #return

        # total presence percentage of every object
        percentages = {"real": {}, "sim": {}}
        for realsim in epoch_counts.keys():
            for key in epoch_counts[realsim].keys():
                percentages[realsim][key] = epoch_counts[realsim][key]/np.sum(list(epoch_counts[realsim].values()))

        if hasattr(self.model, "write_eoe_summaries"):
            self.model.write_eoe_summaries(inference_type, epoch_num)

        print("")

        for key in sorted(epoch_metrics.keys()):
            scalar_name = self.name + "_" + key
            if eval:
                self.model.test_writer.add_scalar(scalar_name, epoch_metrics[key], epoch_num)
            else:
                self.model.train_writer.add_scalar(scalar_name, epoch_metrics[key], epoch_num)

        for realsim in epoch_counts.keys():
            for key in epoch_counts[realsim].keys():
                scalar_name = self.name + "_percentages_" + realsim + "/" + key
                if eval:
                    self.model.test_writer.add_scalar(scalar_name, percentages[realsim][key], epoch_num)
                else:
                    self.model.train_writer.add_scalar(scalar_name, percentages[realsim][key], epoch_num)

        epoch_loss = epoch_metrics["/shared/total_loss"]

        return epoch_loss