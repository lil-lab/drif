import torch
import torch.nn as nn

class WassersteinCriticBase(nn.Module):
    def __init__(self):
        super(WassersteinCriticBase, self).__init__()
        self.wasserstein=True
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.clip_value = None

    def get_iter(self):
        return int(self.iter.data.item())

    def inc_iter(self):
        self.iter += 1

    def clip_weights(self):
        # Clip weights of discriminator
        if self.clip_value:
            for p in self.parameters():
                if p is self.iter:
                    continue
                p.data.clamp_(-self.clip_value, self.clip_value)

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


    def calc_domain_loss(self, real_activation_store, sim_activation_store):
        ...