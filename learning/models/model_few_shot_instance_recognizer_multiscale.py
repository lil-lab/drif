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


class ModelFewShotInstanceRecognizerMultiscale(nn.Module):

    def __init__(self, run_name="", domain="sim"):
        super(ModelFewShotInstanceRecognizerMultiscale, self).__init__()
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
        self.lingunet_encoder = Lingunet5Encoder(lingunet_params)
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
        self.scale_pooling = self.root_params["scale_pooling"]

        #self._attn_k_linear = nn.Linear(self.num_feature_channels, self.attn_dim)
        self._attn_k_linear = nn.ModuleList(
            [nn.Linear(self.num_feature_channels, self.attn_dim) for _ in range(5)])
        self._attn_queries = nn.Parameter(torch.zeros([5, self.attn_dim, self.num_attn_heads]), requires_grad=True)
        torch.nn.init.xavier_uniform(self._attn_queries)

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
        self.lingunet_encoder.init_weights()
        self.lingunet_encoder_s.init_weights()
        self.lingunet_decoder.init_weights()
        self.lingunet_convoob.init_weights()

    def cam_poses_from_states(self, states):
        cam_pos = states[:, 9:12]
        cam_rot = states[:, 12:16]
        pose = Pose(cam_pos, cam_rot)
        return pose

    def _multihead_attention(self, kernels, layer):
        """
        :param kernels: BxKxC tensor (Batch size x Num Keys x Num Channels)
        :return: BxAxC tensor (Batch size x Num Attn Heads x Num Channels)
        """
        bsize = kernels.shape[0]
        # Attention keys: BxKxKdim
        attn_k = self._attn_k_linear[layer](kernels)
        # Attention queries: BxKdimxA
        attn_q = self._attn_queries[layer][np.newaxis, :, :].repeat((bsize, 1, 1))
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

    # TODO: Add a standardized function to create a multi-scale cascade of query images.

    def forward(self, multiscale_query_images, scene_image):
        """
        :param multiscale_query_images: List of BxDxCxHxW batch of D query images for each example.
        :param scene_image: BxCxHxW batch of scene images to locate each of the B queries in.
        :return:
        """
        batch_size = scene_image.shape[0]
        num_scales = len(multiscale_query_images)
        num_queries = multiscale_query_images[0].shape[1]
        num_channels = self.root_params["channels"]
        num_layers = 5

        # Get multiple scale encodings of the query image
        # Combine batch and query image axes into a single batch axis
        multiscale_kernels = []
        for q, query_images in enumerate(multiscale_query_images):
            query_images_multiplex = query_images.view([-1, query_images.shape[2], query_images.shape[3], query_images.shape[4]])
            layer_reprs_m = self.lingunet_encoder(query_images_multiplex)
            layer_reprs_m = [F.avg_pool2d(q, (q.shape[2], q.shape[3])) for q in layer_reprs_m]
            layer_kernels = [q.view([batch_size, num_queries, q.shape[1], q.shape[2], q.shape[3]]) for q in layer_reprs_m]
            layer_kernels = torch.stack(layer_kernels, dim=1)
            multiscale_kernels.append(layer_kernels)

        # multiscale_kernels: list of BxLxQxCx1x1 convolutional kernels, one for each scale
        multiscale_kernels = torch.stack(multiscale_kernels, dim=1)
        # now: multiscale_kernels: BxSxLxQxCx1x1 stack of convolutional kernels (S is # of scales, Q is # of queries)
        #                          0 1 2 3 4 5 6

        # First swap axes to LxBxSxQxCx1x1, so that we have a batch of batched attentions, one per layer
        #                 2 0 1 3 4 5 6
        multiscale_kernels_t_ = multiscale_kernels.permute((2, 0, 1, 3, 4, 5, 6))
        # Then combine batch and scale dimension
        # Reshape each kernel to Lx(BxS)xQxC
        multiscale_kernels_t_ = multiscale_kernels_t_.view([num_layers, batch_size * num_scales, num_queries, num_channels])
        # Compute kernel from multiple queries
        multiscale_attn_kernels_t = [self._multihead_attention(k, i) for i, k in enumerate(multiscale_kernels_t_)]
        multiscale_attn_kernels_t = torch.stack(multiscale_attn_kernels_t, dim=0)
        # multiscale_attn_kernels_t: (Lx(B*S)xAxCx1x1)
        multiscale_attn_kernels = multiscale_attn_kernels_t.view([num_layers, batch_size, self.num_attn_heads * num_scales, num_channels, 1, 1])
        # multiscale_: (LxBx(S*A)xCx1x1
        # Here A*S is the number of feature map output layers

        assert list(multiscale_attn_kernels.shape) == [num_layers, batch_size, num_scales * self.num_attn_heads, num_channels, 1, 1]

        # Encode the scene image
        s1, s2, s3, s4, s5 = self.lingunet_encoder_s(scene_image)

        # Filter the scene image, using the query embeddings as filters
        # At each spatial scale, use the convolutional filter derived from that layer output from the query encoder
        sq1, sq2, sq3, sq4, sq5 = self.lingunet_filter(s1, s2, s3, s4, s5,
                                                       multiscale_attn_kernels[0],
                                                       multiscale_attn_kernels[1],
                                                       multiscale_attn_kernels[2],
                                                       multiscale_attn_kernels[3],
                                                       multiscale_attn_kernels[4])
        # Each sqX is Bx(S*A)xHxW.
        if self.scale_pooling:
            # Reshape to BxSxAxHxW
            sq1, sq2, sq3, sq4, sq5 = [x.view([batch_size, num_scales, self.num_attn_heads, x.shape[2], x.shape[3]])
                                       for x in [sq1, sq2, sq3, sq4, sq5]]
            # Max across scale dimension
            sq1, sq2, sq3, sq4, sq5 = [torch.mean(x, dim=1) for x in (sq1, sq2, sq3, sq4, sq5)]
        # TODO: Add pooling across multiple scales
        sq46, sq37, sq28, sq19, out = self.lingunet_decoder(scene_image, sq1, sq2, sq3, sq4, sq5)

        # Predict probability masses / scores for the goal or trajectory traveling outside the observed part of the map
        o = self.lingunet_convoob(sq19)
        outer_scores = F.avg_pool2d(o, o.shape[2]).view([batch_size, self.root_params["LingUNet"]["out_channels"]])
        both_dist_scores = Partial2DDistribution(out, outer_scores)
        return both_dist_scores

        # TODO: Try another variant of this model that does multi-scale cascading externally, and returns the
        # mask from the spatial scale that has the higherst P(object visible).

    def maybe_cuda(self, tensor):
        return tensor.to(next(self.parameters()).device)

    def cuda_var(self, tensor):
        return tensor.to(next(self.parameters()).device)

    def unbatch(self, batch, halfway=False):
        # Inputs
        scene_images = self.maybe_cuda(batch["scene"])
        mask_labels = self.maybe_cuda(batch["mask"])
        query_images = [self.maybe_cuda(q) for q in batch["query"]]
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
        return FewShotInstanceDataset(dataset_dir=dataset_dir, eval=eval, scales=[12, 16, 24, 36, 54, 81])
