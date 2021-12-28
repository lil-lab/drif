import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_io.paths import get_logging_dir
from learning.datasets.few_shot_instance_dataset import FewShotInstanceDataset
from learning.inputs.pose import Pose
from learning.modules.spatial_softmax_2d import SpatialSoftmax2d
from learning.modules.key_tensor_store import KeyTensorStore
from learning.modules.auxiliary_losses import AuxiliaryLosses
from learning.modules.visitation_softmax import VisitationSoftmax
from learning.meters_and_metrics.instance_segmentation_mask_metrics import InstanceSegmentationMaskMetric

from learning.modules.unet.lingunet_5_instance_det import Lingunet5Encoder, Lingunet5Decoder, Lingunet5Filter, DoubleConv
from learning.inputs.partial_2d_distribution import Partial2DDistribution

from learning.modules.instance_detector.feature_extractor import FeatureExtractor

from utils.simple_profiler import SimpleProfiler
from utils.logging_summary_writer import LoggingSummaryWriter
from learning.meters_and_metrics.meter_server import get_current_meters
from utils.dummy_summary_writer import DummySummaryWriter

import parameters.parameter_server as P

PROFILE = False
# Set this to true to project the RGB image instead of feature map
IMG_DBG = False


class ModelFewShotInstanceRecognizer(nn.Module):

    def __init__(self, run_name="", domain="sim"):
        super(ModelFewShotInstanceRecognizer, self).__init__()
        self.model_name = "few_shot_rec"
        self.run_name = run_name
        self.domain = domain
        self.writer = LoggingSummaryWriter(log_dir=f"{get_logging_dir()}/runs/{run_name}/{self.domain}")

        self.root_params = P.get_current_parameters()["ModelFewShotInstanceRecognizer"]
        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.metric = InstanceSegmentationMaskMetric()

        self.tensor_store = KeyTensorStore()
        self.losses = AuxiliaryLosses()

        self.spatialsoftmax = SpatialSoftmax2d()
        self.visitation_softmax = VisitationSoftmax()

        lingunet_params = self.root_params["LingUNet"]
        self.lingunet_encoder_q = Lingunet5Encoder(lingunet_params)
        self.lingunet_encoder_s = Lingunet5Encoder(lingunet_params)
        self.lingunet_filter = Lingunet5Filter(lingunet_params)
        self.lingunet_decoder = Lingunet5Decoder(lingunet_params)
        self.lingunet_convoob = DoubleConv(
            lingunet_params["hb1"] + lingunet_params["hc3"],
            lingunet_params["out_channels"],
            3, stride=2, padding=1, stride2=2, cmid=16)

        # Auxiliary Objectives
        # --------------------------------------------------------------------------------------------------------------
        # TODO: Add distribution cross-entropy auxiliary objective
        self.losses.print_auxiliary_info()

    def make_picklable(self):
        self.writer = DummySummaryWriter()

    def steal_cross_domain_modules(self, other_self):
        pass

    def both_domain_parameters(self, other_self):
        for p in other_self.parameters():
            yield p
        # Then yield all the parameters from the this module that are not shared with the other one
        for p in self.img_to_features_w.parameters():
            pass
        return

    def get_iter(self):
        return int(self.iter.data[0])

    def inc_iter(self):
        self.iter += 1

    def init_weights(self):
        self.lingunet_encoder_s.init_weights()
        self.lingunet_encoder_q.init_weights()
        self.lingunet_decoder.init_weights()
        self.lingunet_convoob.init_weights()

    def cam_poses_from_states(self, states):
        cam_pos = states[:, 9:12]
        cam_rot = states[:, 12:16]
        pose = Pose(cam_pos, cam_rot)
        return pose

    def forward(self, query_images, scene_image):
        """
        :param query_images: BxDxCxHxW batch of D query images for each example.
        :param scene_image: BxCxHxW batch of scene images to locate each of the B queries in.
        :return:
        """
        batch_size = scene_image.shape[0]

        # Get multiple scale encodings of the query image
        q1, q2, q3, q4, q5 = self.lingunet_encoder_q(query_images[:, 0])
        # Construct a convolutional kernel where the different output layers correspond to different scales:
        q1k = F.avg_pool2d(q1, (q1.shape[2], q1.shape[3]))
        q2k = F.avg_pool2d(q2, (q2.shape[2], q2.shape[3]))
        q3k = F.avg_pool2d(q3, (q3.shape[2], q3.shape[3]))
        q4k = F.avg_pool2d(q4, (q4.shape[2], q4.shape[3]))
        q5k = F.avg_pool2d(q5, (q5.shape[2], q5.shape[3]))
        # Kernel should be batch_size x 5 x channels x 1 x 1
        kernel = torch.stack([q1k, q2k, q3k, q4k, q5k], dim=1)
        assert list(kernel.shape) == [batch_size, 5, q1.shape[1], 1, 1]

        # Encode the scene image
        s1, s2, s3, s4, s5 = self.lingunet_encoder_s(scene_image)

        # Filter the scene image, using the query embeddings as filters
        # Use the same filter bank at all spatial scales to try to get at spatial invariance
        sq1, sq2, sq3, sq4, sq5 = self.lingunet_filter(s1, s2, s3, s4, s5, kernel, kernel, kernel, kernel, kernel)
        sq46, sq37, sq28, sq19, out = self.lingunet_decoder(scene_image, sq1, sq2, sq3, sq4, sq5)

        # Predict probability masses / scores for the goal or trajectory traveling outside the observed part of the map
        o = self.lingunet_convoob(sq19)
        outer_scores = F.avg_pool2d(o, o.shape[2]).view([batch_size, self.root_params["LingUNet"]["out_channels"]])
        both_dist_scores = Partial2DDistribution(out, outer_scores)
        return both_dist_scores

    def maybe_cuda(self, tensor):
        return tensor.to(next(self.parameters()).device)

    def cuda_var(self, tensor):
        return tensor.to(next(self.parameters()).device)

    def unbatch(self, batch, halfway=False):
        # Inputs
        scene_images = self.maybe_cuda(batch["scene"])
        mask_labels = self.maybe_cuda(batch["mask"])
        query_images = self.maybe_cuda(batch["query"])

        # Create a channel dimension (necessary)
        mask_labels = mask_labels[:, np.newaxis, :, :]
        mask_labels = Partial2DDistribution.from_distribution(mask_labels)
        return query_images, scene_images, mask_labels

    # Forward pass for training
    def sup_loss_on_batch(self, batch, eval, halfway=False, grad_noise=False, disable_losses=[]):
        self.prof.tick("out")

        if batch is None:
            print("Skipping None Batch")
            zero = torch.zeros([1]).float().to(next(self.parameters()).device)
            return zero, self.tensor_store

        query_images, scene_images, mask_labels = self.unbatch(batch)
        self.prof.tick("unbatch_inputs")

        # ----------------------------------------------------------------------------
        dist_scores = self(query_images, scene_images)
        # ----------------------------------------------------------------------------
        # The returned values are not used here - they're kept in the tensor store which is used as an input to a loss
        self.prof.tick("call")

        loss = Partial2DDistribution.cross_entropy(dist_scores, mask_labels)

        dist_probs = dist_scores.softmax()

        # Log values, and upload to meters
        self.metric.log_mask(dist_probs, mask_labels)
        self.metric.consolidate()

        #losses, metrics = self.losses.calculate_aux_loss(tensor_store=self.tensor_store, reduce_average=True, disable_losses=disable_losses)
        #loss = self.losses.combine_losses(losses, self.aux_weights)
        #self.writer.add_dict(prefix, losses, iteration)
        #self.writer.add_dict(prefix, metrics, iteration)

        self.prof.tick("calc_losses")

        prefix = self.model_name + ("/eval" if eval else "/train")
        iteration = self.get_iter()
        self.writer.add_dict(prefix, get_current_meters(), iteration)
        self.writer.add_scalar(prefix, loss.item(), iteration)

        self.inc_iter()

        self.prof.tick("summaries")
        self.prof.loop()
        self.prof.print_stats(1)

        return loss, self.tensor_store

    def get_dataset(self, eval=False):
        if eval:
            dataset_dir = P.get_current_parameters()["Data"].get("few_shot_recognizer_test_dataset_dir")
        else:
            dataset_dir = P.get_current_parameters()["Data"].get("few_shot_recognizer_training_dataset_dir")
        return FewShotInstanceDataset(dataset_dir=dataset_dir)
