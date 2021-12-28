import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_io.paths import get_logging_dir
from learning.datasets.region_proposal_or_refinement_dataset import RegionProposalOrRefinementDataset
from learning.modules.key_tensor_store import KeyTensorStore
from learning.modules.auxiliary_losses import AuxiliaryLosses
from learning.models.visualization.viz_html_simple_matching_network import visualize_model
from learning.models.navigation_model_component_base import NavigationModelComponentBase
from learning.modules.unet.simple_unet import SimleUNet
from learning.modules.image_resize import ImageResizer
from learning.modules.generic_model_state import GenericModelState

from utils.simple_profiler import SimpleProfiler
from utils.logging_summary_writer import LoggingSummaryWriter
from learning.meters_and_metrics.meter_server import get_current_meters
from utils.dummy_summary_writer import DummySummaryWriter
from visualization import Presenter

import parameters.parameter_server as P

PROFILE = False
# Set this to true to project the RGB image instead of feature map
IMG_DBG = False


class ModelRegionRefinementNetwork(NavigationModelComponentBase):

    def __init__(self, run_name="", domain="sim", nowriter=False):
        super(ModelRegionRefinementNetwork, self).__init__(run_name, domain, "region_refinement", nowriter)
        self.train_dataset = None
        self.eval_dataset = None

        self.root_params = P.get_current_parameters()["ModelRegionRefinementNetwork"]
        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        #self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.model_state = GenericModelState()

        unet_params = self.root_params["UNet"]
        self.unet = SimleUNet(unet_params)
        self.image_resizer = ImageResizer()

        self.runcount = 0
        self.input_size = 32

    """
    def make_picklable(self):
        self.writer = DummySummaryWriter()

    def steal_cross_domain_modules(self, other_self):
        pass

    def both_domain_parameters(self, other_self):
        for p in other_self.parameters():
            yield p

    def get_iter(self):
        return int(self.iter.data[0])

    def inc_iter(self):
        self.iter += 1
    """

    def init_weights(self):
        self.unet.init_weights()

    def unbatch(self, batch, halfway=False):
        images, mask_gt = batch
        if isinstance(images, list):
            images = [self.to_model_device(img) for img in images]
        else:
            images = self.to_model_device(images)
        if isinstance(mask_gt, list):
            mask_gt = [self.to_model_device(mask) for mask in mask_gt]
        else:
            mask_gt = self.to_model_device(mask_gt)
        return images, mask_gt

    def forward(self, images, logits=False):
        resized_in_dataset = False
        if isinstance(images, torch.Tensor):
            resized_in_dataset = True

        # Images should be a list of differently sized images, resize them to the correct size for prediction
        if not resized_in_dataset:
            keep_sizes = [(img.shape[1], img.shape[2]) for img in images]
            images_s_norm = self.image_resizer.resize_to_target_size_and_normalize(images)
        else:
            images_s_norm = images

        # Run the refinement model
        pred_masks_logits_s = self.unet(images_s_norm)

        # Resize back to input size
        if not resized_in_dataset:
            pred_masks_logits = self.image_resizer.resize_back_from_target_size(pred_masks_logits_s, keep_sizes)
        else:
            pred_masks_logits = pred_masks_logits_s

        return pred_masks_logits if logits else torch.sigmoid(pred_masks_logits)

    # Forward pass for training
    def sup_loss_on_batch(self, batch, eval, halfway=False, grad_noise=False, disable_losses=[]):
        self.prof.tick("out")

        if batch is None:
            print("Skipping None Batch")
            zero = torch.zeros([1]).float().to(next(self.parameters()).device)
            return zero, self.model_state

        images, masks_gt = self.unbatch(batch)
        self.prof.tick("unbatch_inputs")

        # ----------------------------------------------------------------------------
        pred_masks_logits = self(images, logits=True)
        batch_size = pred_masks_logits.shape[0]
        pred_masks_probs = torch.sigmoid(pred_masks_logits)

        loss = F.binary_cross_entropy_with_logits(
            pred_masks_logits.view([batch_size, -1]),
            masks_gt.view([batch_size, -1]))

        if self.runcount % 23 == 0:
            Presenter().show_image(images[0].detach().cpu(), "image", scale=4, waitkey=1)
            Presenter().show_image(masks_gt[0].detach().cpu(), "mask_gt", scale=4, waitkey=1)
            Presenter().show_image(pred_masks_probs[0].detach().cpu(), "mask_pred", scale=4, waitkey=1)

        # ----------------------------------------------------------------------------
        if eval:
            self.runcount += 1
        if eval and self.runcount % 99 == 0:
            visualize_model(images, pred_masks_probs, self.get_iter(), self.run_name)
        # ----------------------------------------------------------------------------
        # The returned values are not used here - they're kept in the tensor store which is used as an input to a loss
        self.prof.tick("call")
        self.prof.tick("calc_losses")

        prefix = self.model_name + ("/eval" if eval else "/train")
        iteration = self.get_iter()
        self.writer.add_dict(prefix, get_current_meters(), iteration)
        self.writer.add_scalar(prefix, loss.item(), iteration)
        self.inc_iter()

        self.prof.tick("summaries")
        self.prof.loop()
        self.prof.print_stats(1)

        return loss, self.model_state

    def _init_dataset(self, eval=False):
        if eval:
            dataset_name_composed = P.get_current_parameters()["Data"].get("rpn_test_dataset_composed")
            dataset_name_raw = P.get_current_parameters()["Data"].get("rpn_test_dataset_raw")
        else:
            dataset_name_composed = P.get_current_parameters()["Data"].get("rpn_training_dataset_composed")
            dataset_name_raw = P.get_current_parameters()["Data"].get("rpn_training_dataset_raw")

        grayscale = P.get_current_parameters()["Data"].get("grayscale", False)
        blur = P.get_current_parameters()["Data"].get("blur", False)
        noise = P.get_current_parameters()["Data"].get("noise", False)
        flip = P.get_current_parameters()["Data"].get("flip", False)
        if grayscale:
            print("Matching network running GRAYSCALE!")
        return RegionProposalOrRefinementDataset(
            composed_dataset_name=dataset_name_composed,
            raw_dataset_name=dataset_name_raw,
            mode="region_refinement",
            eval=eval,
            blur=blur, grain=noise, flip=flip, grayscale=grayscale)

    def get_dataset(self, eval=False):
        return self._init_dataset(eval)
