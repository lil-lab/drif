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

        self.num_attn_heads = self.root_params["attention_heads"]
        self.attn_dim = self.root_params["attn_dim"]
        self.num_feature_channels = lingunet_params["hc1"]
        self._attn_k_linear = nn.Linear(self.num_feature_channels, self.attn_dim)
        self._attn_queries = nn.Parameter(torch.zeros([self.attn_dim, self.num_attn_heads]), requires_grad=True)
        torch.nn.init.xavier_uniform(self._attn_queries)

        self.dropout2d = nn.Dropout2d(p=0.5)
        self.dropout = nn.Dropout(p=0.5)

        """self.mh_attention = torch.nn.MultiheadAttention(
            lingunet_params["hc1"],
            self.root_params["attention_heads"],
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None)"""

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

    def _multihead_attention(self, kernels):
        """
        :param kernels: BxKxC tensor (Batch size x Num Keys x Num Channels)
        :return: BxAxC tensor (Batch size x Num Attn Heads x Num Channels)
        """
        bsize = kernels.shape[0]
        # Attention keys: BxKxKdim
        attn_k = self._attn_k_linear(kernels)
        # Attention queries: BxKdimxA
        attn_q = self._attn_queries[np.newaxis, :, :].repeat((bsize, 1, 1))
        # Attention values: BxKxC
        attn_v = kernels

        # Attention scores matrix: BxKxA
        attn_scores = torch.matmul(attn_k, attn_q)
        # Normalized attn scores matrix: BxKxA (normalize across key/input dimension)
        attn_weights = attn_scores.softmax(dim=1)

        # Attention-weighted inputs: B x K x A x C
        #  values -> B x K x 1 x C
        #  weights -> B x K x A x 1
        attn_weighted_values = attn_v[:, :, np.newaxis, :] * attn_weights[:, :, :, np.newaxis]

        # Output, summing across key dim: B x A x C
        attn_out = attn_weighted_values.sum(1)
        return attn_out

    def forward(self, query_images, scene_image):
        """
        :param query_images: BxDxCxHxW batch of D query images for each example.
        :param scene_image: BxCxHxW batch of scene images to locate each of the B queries in.
        :return:
        """
        batch_size = scene_image.shape[0]
        num_queries = query_images.shape[1]
        num_layers = 5

        # Get multiple scale encodings of the query image
        # Combine batch and query image axes into a single batch axis
        query_images_multiplex = query_images.view([-1, query_images.shape[2], query_images.shape[3], query_images.shape[4]])
        enc_layers_multiplex = self.lingunet_encoder_q(query_images_multiplex)

        # Dropout
        enc_layers_multiplex = [self.dropout2d(e) for e in enc_layers_multiplex]

        # Construct a convolutional kernel where the different output layers correspond to different scales:
        kernels_multiplex = [F.avg_pool2d(e, (e.shape[2], e.shape[3])) for e in enc_layers_multiplex]
        q1k, q2k, q3k, q4k, q5k = [e.view([batch_size, num_queries, e.shape[1], e.shape[2], e.shape[3]]) for e in kernels_multiplex]
        num_channels = q1k.shape[2]

        #q1k = F.avg_pool2d(q1, (q1.shape[2], q1.shape[3]))
        #q2k = F.avg_pool2d(q2, (q2.shape[2], q2.shape[3]))
        #q3k = F.avg_pool2d(q3, (q3.shape[2], q3.shape[3]))
        #q4k = F.avg_pool2d(q4, (q4.shape[2], q4.shape[3]))
        #q5k = F.avg_pool2d(q5, (q5.shape[2], q5.shape[3]))
        # Kernels should be batch_size x 5 (layers) x num_queries x channels x 1 x 1
        query_kernels = torch.stack([q1k, q2k, q3k, q4k, q5k], dim=1)

        # Reshape to (BxL)xQxC
        query_kernels_t_ = query_kernels.view([batch_size * num_layers, num_queries, num_channels])
        # Compute kernel from multiple queries
        attn_kernels_t = self._multihead_attention(query_kernels_t_)
        # attn_kernels_t: (Bx(A*L)xCx1x1
        attn_kernel = attn_kernels_t.view([batch_size, self.num_attn_heads * num_layers, num_channels, 1, 1])
        # Here A*L is the number of feature map output layers

        # Kernel should be batch_size x attn_heads x 5 (layers) x channels x 1 x 1

        # Kernel should be batch_size x 5 (layers) x channels x 1 x 1
        #kernel = query_kernels[:, :, 0, :, :, :]

        assert list(attn_kernel.shape) == [batch_size, num_layers * self.num_attn_heads, q1k.shape[2], 1, 1]

        # Encode the scene image
        s1, s2, s3, s4, s5 = self.lingunet_encoder_s(scene_image)

        # Filter the scene image, using the query embeddings as filters
        # Use the same filter bank at all spatial scales to try to get at spatial invariance
        sq1, sq2, sq3, sq4, sq5 = self.lingunet_filter(s1, s2, s3, s4, s5, attn_kernel, attn_kernel, attn_kernel, attn_kernel, attn_kernel)
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

    def reset_metrics(self):
        self.metric.reset()

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
        grayscale = P.get_current_parameters()["Data"].get("instance_rec_grayscale", False)
        use_all_queries = P.get_current_parameters()["Data"].get("use_all_queries", False)
        if grayscale:
            print("Instance recognizer running GRAYSCALE!")
        return FewShotInstanceDataset(dataset_dir=dataset_dir, eval=eval, blur=True, grain=True,
                                      grayscale=grayscale, use_all_queries=use_all_queries)
