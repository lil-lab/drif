from __future__ import print_function
import torch.nn as nn
import torch
from tensorboardX import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self, run_name, ngpu=1, nc=32, ndf=32, grad_reversal=False):
        super(Discriminator, self).__init__()
        self.model_name = "critic"
        self.writer = SummaryWriter(log_dir="runs/" + run_name)
        self.wasserstein=False
        self.ngpu = ngpu
        self.loss = nn.BCELoss(size_average=True)
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.grad_reversal = grad_reversal
        # Architecture is the first layers of DC-GAN
        self.main = nn.Sequential(
            # input is (nc) x 18 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 9 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 4 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Linear classifier
        self.regressor = nn.Sequential(
            nn.Linear((ndf*4) * 2 * 4, 1),
            nn.Sigmoid())

    def get_iter(self):
        return int(self.iter.data.item())

    def inc_iter(self):
        self.iter += 1

    def forward(self, input):
        if self.grad_reversal:
            input = grad_reverse(input)
        if input.is_cuda and self.ngpu > 1:
            features = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            features = features.view(features.size(0), -1)
            output = self.regressor(features)
        else:
            features = self.main(input)
            features = features.view(features.size(0), -1)
            output = self.regressor(features)

        return output.view(-1, 1).squeeze(1)

    def sup_loss_on_batch(self, features_real, features_sim) -> object:
        batch_size = len(features_sim)

        pred_real = self.forward(features_real.detach())
        pred_sim = self.forward(features_sim.detach())

        real_label = 1.
        sim_label = 0.

        labels_real = torch.full((batch_size,), real_label).cuda()
        labels_sim = torch.full((batch_size,), sim_label).cuda()

        loss_real = self.loss(pred_real, labels_real)
        loss_sim = self.loss(pred_sim, labels_sim)

        # Accuracies are computed for information purpose
        real_classif_accuracy = torch.sum(pred_real > 0.5).type(torch.FloatTensor)/batch_size
        sim_classif_accuracy = torch.sum(pred_sim < 0.5).type(torch.FloatTensor)/batch_size
        classif_accuracy = (sim_classif_accuracy + real_classif_accuracy)/2

        critic_metrics = {"loss_real": loss_real, "loss_sim": loss_sim, "accuracy": classif_accuracy}

        return critic_metrics


class Critic(nn.Module):
    def __init__(self, run_name, ngpu=1, nc=32, ndf=32, improved=False, grad_reversal=False):
        super(Critic, self).__init__()
        self.model_name = "critic"
        self.writer = SummaryWriter(log_dir="runs/" + run_name)
        self.wasserstein=True
        self.ngpu = ngpu
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.grad_reversal = grad_reversal

        # if True, remove batch normalization
        self.improved = improved
        # TODO: try more sophisticated networks.
        # Indeed network cannot be too strong because of Wasserstein GAN property

        if self.improved:
            self.main = nn.Sequential(
                # input is (nc) x 18 x 32
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 9 x 16
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 4 x 8
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.main = nn.Sequential(
                # input is (nc) x 18 x 32
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 9 x 16
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 4 x 8
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.regressor = nn.Linear((ndf*4) * 2 * 4, 1)

    def get_iter(self):
        return int(self.iter.data.item())

    def inc_iter(self):
        self.iter += 1

    def forward(self, input):
        if self.grad_reversal:
            input = grad_reverse(input)
        if input.is_cuda and self.ngpu > 1:
            features = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            features = features.view(features.size(0), -1)
            output = self.regressor(features)
        else:
            features = self.main(input)
            features = features.view(features.size(0), -1)
            output = self.regressor(features)

        return output.view(-1, 1).squeeze(1)

    def calc_gradient_penalty(self, features_real, features_sim):

        # Interpolate real and simulated features. This idea is from improved Wassertein GAN training and add this
        # regularization loss to the total loss.
        # cf https://github.com/jalola/improved-wgan-pytorch/blob/master/gan_train.py
        BATCH_SIZE = len(features_real)
        alpha = torch.rand(BATCH_SIZE, 1)
        alpha = alpha.expand(BATCH_SIZE, int(features_real.nelement() / BATCH_SIZE)).contiguous()
        alpha = alpha.view(features_real.shape)
        alpha = alpha.cuda()

        interpolates = alpha * features_real.detach() + ((1 - alpha) * features_sim.detach())

        interpolates.requires_grad_(True)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_loss = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_loss


    def sup_loss_on_batch(self, features_real, features_sim, lambda_reg=0):

        pred_real = self.forward(features_real.detach())
        pred_sim = self.forward(features_sim.detach())

        # cf Wasserstein GAN paper. The critic tries to maximize this difference.
        loss_wass = torch.mean(pred_real) - torch.mean(pred_sim)

        critic_metrics = {"wass_loss": loss_wass}
        total_loss = loss_wass
        if lambda_reg > 0:
            gradient_loss = self.calc_gradient_penalty(features_real, features_sim)
            critic_metrics["gradient_loss"] = gradient_loss
            total_loss += lambda_reg * gradient_loss
        critic_metrics["total_loss"] = total_loss

        return critic_metrics


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)