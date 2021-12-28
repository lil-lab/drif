import sys
import os
import ray
import functools
import copy
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from data_io.instructions import get_all_instructions
from data_io.instructions import get_word_to_token_map
from utils.simple_profiler import SimpleProfiler
from learning.utils import get_n_params, get_n_trainable_params

import parameters.parameter_server as P

PROFILE = False


def get_model_parameters(model, named=False):
    params_out = []
    skipped_params = 0
    if named:
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_out.append((name, param))
            else:
                skipped_params += 1
    else:
        for param in model.parameters():
            if param.requires_grad:
                params_out.append(param)
            else:
                skipped_params += 1
    return params_out


class ForwardBackward:
    def __init__(self, model, device):
        self.device = device
        self.model = model.to(device)
        self.last_loss = None

    def sup_loss_on_batch(self, batch, eval):
        out_loss, model_state = self.model.sup_loss_on_batch(batch, eval)
        self.last_loss = out_loss
        # Detach so that we don't have to send the entire graph between processes
        return out_loss.detach().cpu()

    def backward(self):
        self.last_loss.backward()
        gradients = [param.grad for param in get_model_parameters(self.model)]
        # We won't do double backward, so detach the computational graph:
        gradients = [grad if grad is None else grad.detach().cpu().clone() for grad in gradients]
        # Delete gradients afterwards:
        for param in get_model_parameters(self.model):
            param.grad = None
        return gradients

    def forward_backward(self, batch, eval):
        if eval:
            self.model.eval()
        else:
            self.model.train()
        loss = self.sup_loss_on_batch(batch, eval)
        if not eval:
            with torch.no_grad():
                gradients = self.backward()
        else:
            # Return dummy values if during evaluation
            gradients = [None for _ in get_model_parameters(self.model)]
        return loss, gradients

    def update_model_parameters(self, new_parameters):
        for mparam, nparam in zip(get_model_parameters(self.model), new_parameters):
            # Disable computational graph for now, otherwise the model parameters would become non-leaf variables
            with torch.no_grad():
                mparam.copy_(nparam.to(self.device))

    def call_on_model(self, call_name, *args, **kwargs):
        getattr(self.model, call_name)(*args, **kwargs)


@ray.remote(num_cpus=1, num_gpus=0.5)
class ForwardBackwardActor(ForwardBackward):
    def __init__(self, model, setup_name):
        # The experiment config does not get carried across to different Ray processes, so need to re-load the config.
        P.initialize_experiment(setup_name)
        # Ray has set the CUDA_VISIBLE_DEVICES environment variable for this actor, so all we have to do is call .cuda
        #  and then find out which device the model was actually moved to
        model = model.cuda()
        device = next(model.parameters()).device
        super().__init__(model, device)


class TrainerDataparallel:
    def __init__(
            self,
            model,
            epoch=0,
            name="",
            run_name="",
    ):
        _, _, _, corpus = get_all_instructions()
        self.token2word, self.word2token = get_word_to_token_map(corpus)

        self.params = P.get("Training")
        self.batch_size = self.params['batch_size']
        self.weight_decay = self.params['weight_decay']
        self.optimizer_type = self.params['optimizer']
        self.num_loaders = self.params['num_loaders']
        self.lr = self.params['lr']
        self.dataparallel_workers = self.params['num_dataparallel_workers']
        self.dp_local_device = self.params['dataparallel_local_device']

        self.name = name
        self.dataset_names = None

        n_params = get_n_params(model)
        n_params_tr = get_n_trainable_params(model)
        print("Training Model:")
        print("Number of model parameters: " + str(n_params))
        print("Trainable model parameters: " + str(n_params_tr))

        self.model = model
        self.run_name = run_name
        if self.optimizer_type == "adam":
            make_optimizer = functools.partial(optim.Adam, lr=self.lr, weight_decay=self.weight_decay)
            #self.optim = optim.Adam(get_model_parameters(self.model), self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == "sgd":
            make_optimizer = functools.partial(optim.SGD, lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
            #self.optim = optim.SGD(get_model_parameters(self.model), self.lr, weight_decay=self.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")
        self.optim = make_optimizer(get_model_parameters(self.model))

        self.model.make_picklable()
        setup_name = P.get_setup_name()
        self.worker_actors = [ForwardBackwardActor.remote(self.model, setup_name)
                              for i in range(self.dataparallel_workers)]
        self.model.enable_logging()
        self.local_actor = ForwardBackward(self.model, self.dp_local_device)

        self.train_epoch_num = epoch
        self.train_segment = 0
        self.test_epoch_num = epoch
        self.test_segment = 0
        self.batch_num = 0

    def set_dataset_names(self, dataset_names):
        self.dataset_names = dataset_names

    def set_start_epoch(self, epoch):
        self.train_epoch_num = epoch
        self.test_epoch_num = epoch

    def add_gradients(self, model, new_gradient_list, device):
        for param, grad in zip(get_model_parameters(model), new_gradient_list):
            if grad is not None:
                if param.grad is None:
                    param.grad = grad.to(self.dp_local_device)
                else:
                    param.grad += grad.to(self.dp_local_device)

    def train_epoch(self, train_data=None, train_envs=None, eval=False):
        if eval:
            self.model.eval()
            inference_type = "eval"
            epoch_num = self.train_epoch_num
            for actor in self.worker_actors:
                actor.call_on_model.remote("eval")
            self.test_epoch_num += 1
        else:
            self.model.train()
            inference_type = "train"
            epoch_num = self.train_epoch_num
            for actor in self.worker_actors:
                actor.call_on_model.remote("train")
            self.train_epoch_num += 1

        if hasattr(self.model, "reset_metrics"):
            self.model.reset_metrics()

        if self.dataset_names is None:
            dataset_names = P.get("Data::dataset_names")
        else:
            dataset_names = self.dataset_names

        print("WARNING: ASSUMING THAT SUPERVISED SINGLE_DOMAIN DATA COMES FROM SIMULATOR!")
        try:
            dataset = self.model.get_dataset(data=train_data, envs=train_envs, domain="sim",
                                             dataset_names=dataset_names, dataset_prefix="supervised", eval=eval)
        except Exception as e:
            dataset = self.model.get_dataset(eval=eval)
        # TODO: Get rid of this:
        if hasattr(dataset, "set_word2token"):
            dataset.set_word2token(self.token2word, self.word2token)

        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_loaders,
            pin_memory=False,
            timeout=0,
            drop_last=False)

        num_samples = len(dataset)
        if num_samples == 0:
            print ("DATASET HAS NO DATA!")
            return -1.0

        num_batches = int((num_samples + self.batch_size - 1) / self.batch_size)

        epoch_loss = 0
        count = 0

        prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

        for batch in dataloader:
            # batch is a list of examples. Each example is a dictionary with a bunch of data
            prof.tick("batch_load")

            # Zero gradients
            self.optim.zero_grad()

            # Run N workers in different processes
            losses_and_gradients = [actor.forward_backward.remote(batch_i, eval) for actor, batch_i in
                      zip(self.worker_actors, batch[1:])]
            # Run one worker in the current process
            local_loss, local_grad = self.local_actor.forward_backward(batch[0], eval)

            # Collect the losses and gradients
            losses_and_gradients = ray.get(losses_and_gradients)
            losses = [local_loss] + [lg[0] for lg in losses_and_gradients]
            gradients = [lg[1] for lg in losses_and_gradients]

            # losses is a list, so use the default python sum instead of pytorch
            batch_loss = sum(losses) / (len(losses) + 1e-30)

            prof.tick("forward-backward")

            if not eval:
                # Add all the gradients to the model parameter gradients
                with torch.no_grad():
                    for gradient_list in gradients:
                        self.add_gradients(self.model, gradient_list, self.dp_local_device)
                self.optim.step()

            self.batch_num += 1
            self.optim.zero_grad()

            prof.tick("optim")

            for actor in self.worker_actors:
                actor.update_model_parameters.remote(get_model_parameters(self.model))
            prof.tick("send_params")

            # Get losses as floats
            epoch_loss += batch_loss.data.item()
            count += 1

            sys.stdout.write(
                "\r Batch:" + str(count) + " / " + str(num_batches) + " loss: " + str(batch_loss.data.item()))
            sys.stdout.flush()

            self.train_segment += 0 if eval else 1
            self.test_segment += 1 if eval else 0

            prof.tick("rep")
            prof.loop()
            prof.print_stats(10)

        if hasattr(self.model, "write_eoe_summaries"):
            self.model.write_eoe_summaries(inference_type, epoch_num)

        print("")
        epoch_loss /= (count + 1e-15)

        if hasattr(self.model, "writer"):
            self.model.writer.add_scalar(self.name + "/" + inference_type + "_epoch_loss", epoch_loss, epoch_num)

        return epoch_loss
