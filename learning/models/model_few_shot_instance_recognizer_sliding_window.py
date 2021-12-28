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
from learning.modules.resnet.resnet_13_light import ResNet13Light
from learning.meters_and_metrics.instance_segmentation_mask_metrics import InstanceSegmentationMaskMetric
from learning.meters_and_metrics.binary_classification_metrics import BinaryClassificationMetrics

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

torch.autograd.set_detect_anomaly(True)


class InstanceRecognizerReconstructor(nn.Module):
    def __init__(self, window_scales, window_stride):
        super(InstanceRecognizerReconstructor, self).__init__()

        back_masks_np = FewShotInstanceDataset.compute_window_back_masks(96, 128, window_scales, window_stride)
        self.back_masks = [torch.from_numpy(b) for b in back_masks_np]

    def forward(self, similarity_images):
        """
        :param similarity_images: List of 4 images, each BxYxX, where Y and X are number of vertical and horizontal windows
        :return:
        """
        batch_size = similarity_images[0].shape[0]
        score_image_out = torch.zeros((batch_size, len(similarity_images), 96, 128), device=similarity_images[0].device)
        score_image_out.fill_(0.0)
        if self.back_masks[0].device != similarity_images[0].device:
            self.back_masks = [b.to(similarity_images[0].device) for b in self.back_masks]

        for s, (similarity_image, back_mask) in enumerate(zip(similarity_images, self.back_masks)):
            count_image = torch.zeros((batch_size, back_mask.shape[0], back_mask.shape[1], 96, 128), device=similarity_image.device)
            score_accumulator = torch.zeros((batch_size, back_mask.shape[0], back_mask.shape[1], 96, 128), device=similarity_image.device)
            # TODO: Speed this sh*t up
            #  Stack the back_mask once, so that instead of looping over y and x, we can just do:
            #  count_image[:, back_mask] = 1.0
            #  score_accumulator[:, back_mask] = similarity_image[:, 0:1].view([batch_size, 1, -1])
            #   ^- this isn't quite gonna work due to shape mismatches. Figure it out.
            for y in range(back_mask.shape[0]):
                for x in range(back_mask.shape[1]):
                    count_image[:, y, x, back_mask[y, x]] = 1.0
                    score_accumulator[:, y, x, back_mask[y, x]] = similarity_image[:, 0:1, y, x]
            count_image = count_image.sum(1).sum(1)
            score_accumulator = score_accumulator.sum(1).sum(1)
            score_image_out[:, s] = score_accumulator / (count_image + 1e-30)
        return score_image_out


class ConvEncoderDecoder(nn.Module):
    def __init__(self, mid_channels):
        super(ConvEncoderDecoder, self).__init__()

        self.enc1 = nn.Conv2d(7, mid_channels, 3, 2)
        self.norm1 = nn.InstanceNorm2d(mid_channels)
        self.enc2 = nn.Conv2d(mid_channels, mid_channels, 3, 2)
        self.norm2 = nn.InstanceNorm2d(mid_channels)
        self.enc3 = nn.Conv2d(mid_channels, mid_channels, 3, 2)
        self.enc4 = nn.Conv2d(mid_channels, mid_channels, 3, 2)
        self.enc5 = nn.Conv2d(mid_channels, mid_channels, 3, 2)

        self.dec1 = nn.ConvTranspose2d(mid_channels + mid_channels, 1, 3, 2)
        self.dec2 = nn.ConvTranspose2d(mid_channels + mid_channels, mid_channels, 3, 2)
        self.dnorm2 = nn.InstanceNorm2d(mid_channels)
        self.dec3 = nn.ConvTranspose2d(mid_channels + mid_channels, mid_channels, 3, 2)
        self.dnorm3 = nn.InstanceNorm2d(mid_channels)
        self.dec4 = nn.ConvTranspose2d(mid_channels + mid_channels, mid_channels, 3, 2)
        self.dec5 = nn.ConvTranspose2d(mid_channels, mid_channels, 3, 2)

        self.outer_prob_linear = nn.Linear(2 * 3 * mid_channels, 1)

        self.act = nn.LeakyReLU()

    def forward(self, scene_image, score_image):
        """
        :param scene_image: Bx3xHxW RGB image batch
        :param score_image: Bx4xHxW similarity score image batch
        :return: Partial2dDistribution of size Bx1xHxW
        """
        out_shape = list(scene_image.shape)
        out_shape[1] = 1

        batch_size = scene_image.shape[0]
        x = torch.cat([scene_image, score_image], dim=1)
        f1 = self.norm1(self.act(self.enc1(x)))
        f2 = self.norm2(self.act(self.enc2(f1)))
        f3 = self.act(self.enc3(f2))
        f4 = self.act(self.enc4(f3))
        f5 = self.act(self.enc5(f4))

        df4 = self.act(self.dec5(f5))
        df3 = self.act(self.dec4(torch.cat([df4, f4], dim=1)))
        df2 = self.act(self.dec3(torch.cat([df3, f3], dim=1)))
        df1 = self.act(self.dec2(torch.cat([df2, f2], dim=1)))
        inner_scores = self.dec1(torch.cat([df1, f1], dim=1), output_size=out_shape)

        outer_score = self.outer_prob_linear(f5.view([batch_size, -1]))
        partial_scores = Partial2DDistribution(inner_scores, outer_score)
        return partial_scores


class ModelFewShotInstanceRecognizerSlidingWindow(nn.Module):

    def __init__(self, run_name="", domain="sim"):
        super(ModelFewShotInstanceRecognizerSlidingWindow, self).__init__()
        self.model_name = "few_shot_rec"
        self.run_name = run_name
        self.domain = domain
        self.writer = LoggingSummaryWriter(log_dir=f"{get_logging_dir()}/runs/{run_name}/{self.domain}")

        self.root_params = P.get_current_parameters()["ModelFewShotInstanceRecognizer"]
        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.metric = InstanceSegmentationMaskMetric(reset_every_n=100)
        self.window_metric = BinaryClassificationMetrics(reset_every_n=100)

        self.tensor_store = KeyTensorStore()
        self.losses = AuxiliaryLosses()

        self.spatialsoftmax = SpatialSoftmax2d()
        self.visitation_softmax = VisitationSoftmax()

        self.window_scales = (12, 24, 48, 96)
        self.window_stride = 12
        self.window_size = 32
        self.query_size = 32
        self.window_vector_dim = 32

        self.encoder = ResNet13Light(32)

        self.reconstructor = InstanceRecognizerReconstructor(self.window_scales, self.window_stride)
        self.refiner = ConvEncoderDecoder(32)

        self.bceloss = torch.nn.BCEWithLogitsLoss()

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
        self.encoder.init_weights()
        pass

    def cam_poses_from_states(self, states):
        cam_pos = states[:, 9:12]
        cam_rot = states[:, 12:16]
        pose = Pose(cam_pos, cam_rot)
        return pose

    def extract_windows(self, scene_image):
        """
        :param scene_image: BxCxHxW batch of images
        :return: BxWxCxHxW batch of
        """
        pass

    def similarity(self, query_vectors, scene_window_vectors):
        """
        :param query_vectors: BxQxC - B-sized batch of Q C-dimensional vectors.
        :param scene_window_vectors: BxNxC - B-sized stack
        :return:
        """
        q = query_vectors
        s = scene_window_vectors.transpose(1, 2)
        similarity_matrix = torch.bmm(q, s)
        return similarity_matrix

    def forward(self, query_images, scene_windows):
        """
        :param query_images: BxQxCxHxW batch of Q query images for each example.
        :param scene_windows: list (over scales) of BxYxXxCxHxW batch of scene windows, where X and Y are number of
                                windows vertically and horizontally for that specific scale.
        :return:
        """
        similarity_score_images = self.calc_similarity_score_images(query_images, scene_windows)
        probability_mask = self.calc_probability_mask(similarity_score_images)
        return probability_mask

    def calc_probability_mask(self, scene_image, similarity_score_images):
        """
        :param similarity_score_images: List of BxYxX tensors that represent object match scores at each spatial loc.
        :return: Partial2DDistribution where inner_distribution has shape BxHxW
        """
        score_mask = self.reconstructor(similarity_score_images)
        partial_scores = self.refiner(scene_image, score_mask)
        return partial_scores

    def calc_similarity_score_images(self, query_images, scene_windows):
        """
        :param query_images: BxQxCxHxW batch of Q query images for each example.
        :param scene_windows: list (over scales) of BxYxXxCxHxW batch of scene windows, where X and Y are number of
                                windows vertically and horizontally for that specific scale.
        :return:
        """
        # TODO: Maybe add list of scales as parameter (though )
        batch_size = query_images.shape[0]
        num_queries = query_images.shape[1]

        # Encode each query image into a vector
        stacked_query_images = query_images.view(batch_size * num_queries, 3, self.query_size, self.query_size)
        stacked_query_vectors = self.encoder(stacked_query_images)
        stacked_query_vectors = stacked_query_vectors.view(batch_size, num_queries, self.window_vector_dim)

        # Encode all sliding windows into vectors
        similiarity_score_images = []
        for scale, scale_windows in zip(self.window_scales, scene_windows):
            y = scale_windows.shape[1]
            x = scale_windows.shape[2]
            # Combine window X,Y and batch axes into a single, temporary batch axis
            stacked_windows = scale_windows.view(
                (batch_size * y * x, 3, self.window_size, self.window_size))
            stacked_window_vectors = self.encoder(stacked_windows)
            stacked_window_vectors = stacked_window_vectors.view(batch_size, y * x, self.window_vector_dim)
            stacked_similarity_scores = self.similarity(stacked_query_vectors, stacked_window_vectors)
            similarity_image = stacked_similarity_scores.view([batch_size, num_queries, 1, y, x])
            # similarity_image: BxQxYxX
            # Average across all queries
            # TODO: Consider different types of pooling.
            #  Currently we mix mean and max to have max-like behavior, but without cutting off gradients
            pooled_similarity_image = similarity_image.mean(1) + similarity_image.max(1).values
            similiarity_score_images.append(pooled_similarity_image)

        return similiarity_score_images

    def maybe_cuda(self, tensor):
        return tensor.to(next(self.parameters()).device)

    def cuda_var(self, tensor):
        return tensor.to(next(self.parameters()).device)

    def unbatch(self, batch, halfway=False):
        # Inputs
        scene_images = self.maybe_cuda(batch["scene"])
        mask_labels = self.maybe_cuda(batch["mask"])
        query_images = self.maybe_cuda(batch["query"])

        scene_windows = [self.maybe_cuda(s) for s in batch["scene_windows"]]
        mask_windows = [self.maybe_cuda(s) for s in batch["mask_windows"]]
        window_labels = [self.maybe_cuda(s) for s in batch["window_labels"]]

        # Create a channel dimension (necessary)
        mask_labels = mask_labels[:, np.newaxis, :, :]
        mask_labels = Partial2DDistribution.from_distribution(mask_labels)
        return query_images, scene_images, mask_labels, scene_windows, mask_windows, window_labels

    def run_on_example(self, example):
        print("ding")

    # Forward pass for training
    def sup_loss_on_batch(self, batch, eval, halfway=False, grad_noise=False, disable_losses=[]):
        self.prof.tick("out")

        if batch is None:
            print("Skipping None Batch")
            zero = torch.zeros([1]).float().to(next(self.parameters()).device)
            return zero, self.tensor_store

        query_images, scene_images, mask_labels, scene_windows, mask_windows, window_labels = self.unbatch(batch)
        self.prof.tick("unbatch_inputs")

        # ----------------------------------------------------------------------------
        similarity_score_images = self.calc_similarity_score_images(query_images, scene_windows)
        score_masks = self.calc_probability_mask(scene_images, similarity_score_images)
        #dist_scores = self(query_images, scene_windows)
        # ----------------------------------------------------------------------------
        self.prof.tick("call")

        # Window-wise stuff (stage 1)
        # BCE loss
        flat_scores = torch.cat([s.view([-1]) for s in similarity_score_images], dim=0)
        flat_labels = torch.cat([s.view([-1]) for s in window_labels], dim=0).float()
        bce_loss = self.bceloss(flat_scores, flat_labels)
        # Window-wise metrics
        for s, (similarity_score_image, window_label) in enumerate(zip(similarity_score_images, window_labels)):
            self.window_metric.log_predictions(similarity_score_image.view([-1]), window_label.view([-1]), name=f"scale_{s}")

        # Mask stuff (stage 2)
        ce_loss = Partial2DDistribution.cross_entropy(score_masks, mask_labels)
        dist_probs = score_masks.softmax()
        # Log values, and upload to meters
        self.metric.log_mask(dist_probs, mask_labels)

        loss = ce_loss + bce_loss
        self.prof.tick("calc_losses")

        # Consolidate metrics and store results
        self.metric.consolidate()
        self.window_metric.consolidate()

        prefix = self.model_name + ("/eval" if eval else "/train")
        iteration = self.get_iter()
        self.writer.add_dict(prefix, get_current_meters(), iteration)
        self.writer.add_scalar(f"{prefix}/loss", loss.item(), iteration)
        self.writer.add_scalar(f"{prefix}/bce_loss", bce_loss.item(), iteration)
        self.writer.add_scalar(f"{prefix}/ce_loss", ce_loss.item(), iteration)

        self.inc_iter()
        self.prof.tick("summaries")
        self.prof.loop()
        self.prof.print_stats(1)

        return loss, self.tensor_store

    def reset_metrics(self):
        self.metric.reset()
        self.window_metric.reset()

    def get_dataset(self, eval=False):
        if eval:
            dataset_dir = P.get_current_parameters()["Data"].get("few_shot_recognizer_test_dataset_dir")
        else:
            dataset_dir = P.get_current_parameters()["Data"].get("few_shot_recognizer_training_dataset_dir")
        return FewShotInstanceDataset(dataset_dir=dataset_dir,
                                      eval=eval,
                                      query_scales=(self.query_size, ),
                                      sliding_window=True,
                                      sliding_window_size=self.window_size,
                                      sliding_window_stride=self.window_stride,
                                      window_scales=self.window_scales,
                                      grain=True,
                                      blur=True)
