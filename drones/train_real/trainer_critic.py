import sys
import torch.optim as optim
import torch

import parameters.parameter_server as P


class CriticTrainer():
    def __init__(self, dataloader, model, epoch=0):
        self.epoch = epoch
        if (epoch < 25) | (epoch % 100 == 0):
            self.n_critic = 100
        else:
            self.n_critic = 5

        self.model = model
        self.dataloader = dataloader
        self.params = P.get_current_parameters()["CriticOptimizer"]
        self.batch_size = self.params['batch_size']
        self.weight_decay = self.params['weight_decay']
        self.optimizer = self.params['optimizer']
        self.num_loaders = self.params['num_loaders']
        self.clamp_lower = - self.params["clip"]
        self.clamp_upper = self.params["clip"]
        self.lr = self.params['lr']
        self.lambda_reg = self.params['lambda_reg']
        self.betas = self.params["betas"]

        if self.optimizer == "adam":
            self.optim = optim.Adam(self.get_model_parameters(self.model.discriminator), self.lr,
                                    weight_decay=self.weight_decay)
            if self.model.discriminator.improved:
                self.optim.betas = self.betas

        elif self.optimizer == "sgd":
            self.optim = optim.SGD(self.get_model_parameters(self.model.discriminator), self.lr,
                                   weight_decay=self.weight_decay, momentum=0.9)
        elif self.optimizer == "rmsprop":
            self.optim = optim.RMSprop(self.get_model_parameters(self.model.discriminator), self.lr,
                                   weight_decay=self.weight_decay)

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

    def inc_epoch(self):
        self.epoch += 1
        if self.epoch >= 25:
            self.n_critic = 5
        elif self.epoch % 100 == 0:
            self.n_critic = 100

    def train_epoch(self, i):
        # TODO: Find out why we clamp discriminator parameters
        if not self.model.discriminator.improved:
            for p in self.model.discriminator.parameters():
                p.data.clamp_(self.clamp_lower, self.clamp_upper)

        i_critic = 0
        data_iter = iter(self.dataloader)

        # Train critic network
        while (i_critic < self.n_critic) & (i < len(self.dataloader)):
            batch_critic = data_iter.next()
            self.optim.zero_grad()

            if i_critic == self.n_critic:
                break
            images_real = batch_critic["real"]["images"]
            images_sim = batch_critic["sim"]["images"]

            if self.model.is_cuda:
                images_real = images_real.cuda()
                images_sim = images_sim.cuda()

            # compute features of images
            features_real = self.model.model_real(images_real)
            features_sim = self.model.model_sim(images_sim)

            use_reg = self.model.discriminator.improved
            critic_metrics = self.model.discriminator.sup_loss_on_batch(features_real, features_sim,
                                                                        lambda_reg=self.lambda_reg * use_reg)
            critic_loss = critic_metrics["total_loss"]
            m_one = torch.FloatTensor([-1])
            m_one = m_one.cuda()
            critic_loss.reshape(1).backward(m_one)
            self.optim.step()

            sys.stdout.write(
                "\r Batch:" + str(i) + " / " + str() + ": critic training")  #+ " loss: " + str(batch_loss.data[0]))
            sys.stdout.flush()

            i_critic += 1
            i += 1

        return i, critic_metrics