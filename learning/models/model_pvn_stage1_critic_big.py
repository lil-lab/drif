import torch
import torch.nn as nn

from data_io.paths import get_logging_dir
from learning.inputs.common import empty_float_tensor, cuda_var
from learning.models.wasserstein_critic_base import WassersteinCriticBase
from learning.meters_and_metrics.moving_average import MovingAverageMeter
from learning.modules.key_tensor_store import KeyTensorStore

from visualization import Presenter

from parameters.parameter_server import get_current_parameters
from utils.logging_summary_writer import LoggingSummaryWriter

PROFILE = False
# Set this to true to project the RGB image instead of feature map
IMG_DBG = False


class PVN_Stage1_Critic_Big(WassersteinCriticBase):

    def __init__(self, run_name=""):

        super(PVN_Stage1_Critic_Big, self).__init__()
        self.model_name = "pvn_stage1_critic_big"
        self.run_name = run_name
        self.writer = LoggingSummaryWriter(log_dir=f"{get_logging_dir()}/runs/{run_name}/pvn_stage1_critic")

        self.params = get_current_parameters()["ModelCritic"]
        nc = self.params["feature_channels"]
        ndf = self.params["critic_channels"]
        self.grad_penalty_weight = self.params["grad_penalty_weight"]
        self.clip_value = self.params["clip_value"]

        # if True, remove batch normalization
        self.improved = True
        # TODO: try more sophisticated networks.
        # Indeed network cannot be too strong because of Wasserstein GAN property

        self.main_fpv = nn.Sequential(
            # input is (nc) x 18 x 32
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 9 x 16
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 4 x 8
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 2 x 4
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 1 x 2
        )

        self.main_lingunet = LingUNetActivationEncoder(get_current_parameters()["ModelPVN"]["Stage1"]["lingunet"])

        self.regressor = nn.Linear((ndf * 4) * 2, 1)

        self.goal_acc_meter = MovingAverageMeter(10)

    def init_weights(self):
        pass

    def forward(self, fpv_features, unet_features):
        fpv_features_f = self.main_fpv(fpv_features)
        fpv_features_f = fpv_features_f.view(fpv_features.size(0), -1)

        unet_features_f = self.main_lingunet(unet_features)
        unet_features_f = unet_features_f.view(unet_features.size(0), -1)

        comb_features = torch.cat([fpv_features_f, unet_features_f], 1)

        output = self.regressor(comb_features)
        return output.view(-1, 1).squeeze(1)

    def cuda_var(self, tensor):
        return tensor.to(next(self.parameters()).device)

    # Forward pass for training (with batch optimizations)
    def calc_domain_loss(self, real_activation_store, sim_activation_store):

        tensor_store = KeyTensorStore()

        features_real = real_activation_store.get_inputs_batch("fpv_features", cat_not_stack=True)
        features_sim = sim_activation_store.get_inputs_batch("fpv_features", cat_not_stack=True)

        # Real and simulated features might have different trajectory lengths. This could give away the source domain.
        # To deal with this, randomly sample a subset of the longest trajectory.

        # Use weight clipping instead of gradient penalty
        if self.grad_penalty_weight <= 0:
            self.clip_weights()

        # TODO: Handle different length sequences - choose number of feature maps to randomly sample
        pred_real = self(features_real)
        pred_sim = self(features_sim)

        # cf Wasserstein GAN paper. The critic tries to maximize this difference.
        loss_wass = torch.mean(pred_real) - torch.mean(pred_sim)

        tensor_store.keep_input("wass_loss", loss_wass)

        prefix = "pvn_critic" + ("/eval" if eval else "/train")
        self.writer.add_scalar(f"{prefix}/wass_loss", loss_wass.item(), self.get_iter())

        total_loss = loss_wass
        if self.grad_penalty_weight > 0:
            gradient_loss = self.calc_gradient_penalty(features_real, features_sim)
            tensor_store.keep_input("gradient_loss", gradient_loss)
            total_loss += self.grad_penalty_weight * gradient_loss

        tensor_store.keep_input("total_loss", total_loss)
        self.writer.add_scalar(f"{prefix}/wass_loss_with_penalty", loss_wass.item(), self.get_iter())

        self.inc_iter()

        return total_loss, tensor_store
