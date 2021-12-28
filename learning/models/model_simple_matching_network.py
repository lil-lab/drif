import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from data_io.paths import get_logging_dir
from learning.datasets.object_matching_dataset import ObjectMatchingDataset
from learning.inputs.pose import Pose
from learning.modules.key_tensor_store import KeyTensorStore
from learning.modules.auxiliary_losses import AuxiliaryLosses
from learning.models.visualization.viz_html_simple_matching_network import visualize_model
from learning.modules.unet.lingunet_5_instance_det import Lingunet5Encoder

from learning.modules.instance_detector.feature_extractor import FeatureExtractor

from utils.simple_profiler import SimpleProfiler
from utils.logging_summary_writer import LoggingSummaryWriter
from learning.meters_and_metrics.meter_server import get_current_meters
from utils.dummy_summary_writer import DummySummaryWriter
from utils.dict_tools import objectview

import parameters.parameter_server as P

PROFILE = False
# Set this to true to project the RGB image instead of feature map
IMG_DBG = False


class DoubleConv(torch.nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, cmid=None, stride2=1):
        super(DoubleConv, self).__init__()
        if cmid is None:
            cmid = int((cin + cout) / 2)
        self.conv1 = nn.Conv2d(cin, cmid, k, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(cmid, cout, k, stride=stride2, padding=padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, img):
        x = self.conv1(img)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.p = objectview(params)

        # inchannels, outchannels, kernel size
        self.conv1 = nn.Conv2d(self.p.in_channels, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv2 = nn.Conv2d(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv3 = nn.Conv2d(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv4 = nn.Conv2d(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv5 = nn.Conv2d(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.norm2 = nn.InstanceNorm2d(self.p.hc1)
        self.norm3 = nn.InstanceNorm2d(self.p.hc1)
        self.norm4 = nn.InstanceNorm2d(self.p.hc1)
        self.norm5 = nn.InstanceNorm2d(self.p.hc1)
        self.act = nn.LeakyReLU()

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv3.weight)
        self.conv3.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv4.weight)
        self.conv4.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv5.weight)
        self.conv5.bias.data.fill_(0)

    def forward(self, img):
        x1 = self.norm2(self.act(self.conv1(img)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.act(self.conv3(x2))
        x4 = self.act(self.conv4(x3))
        x5 = self.act(self.conv5(x4))
        return x1, x2, x3, x4, x5


class WideEncoder(torch.nn.Module):
    def __init__(self):
        super(WideEncoder, self).__init__()
        c_a = 32
        c_b = 48
        c_c = 64

        # inchannels, outchannels, kernel size
        self.conv1 = nn.Conv2d(3, c_a, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(c_a, c_a, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(c_a, c_b, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(c_b, c_b, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(c_b, c_c, 3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(c_a)
        self.norm3 = nn.InstanceNorm2d(c_a)
        self.norm4 = nn.InstanceNorm2d(c_b)
        self.norm5 = nn.InstanceNorm2d(c_b)
        self.act = nn.LeakyReLU()

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv3.weight)
        self.conv3.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv4.weight)
        self.conv4.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv5.weight)
        self.conv5.bias.data.fill_(0)

    def forward(self, img):
        x1 = self.norm2(self.act(self.conv1(img)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.act(self.conv3(x2))
        x4 = self.act(self.conv4(x3))
        x5 = self.act(self.conv5(x4))
        return x1, x2, x3, x4, x5


class ModelSimpleMatchingNetwork(nn.Module):

    def __init__(self, run_name="", domain="sim"):
        super(ModelSimpleMatchingNetwork, self).__init__()
        self.model_name = "simple_matching"
        self.run_name = run_name
        self.domain = domain
        self.writer = LoggingSummaryWriter(log_dir=f"{get_logging_dir()}/runs/{run_name}/{self.domain}")
        self.train_dataset = None
        self.eval_dataset = None

        self.root_params = P.get_current_parameters()["ModelSimpleMatchingNetwork"]
        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.tensor_store = KeyTensorStore()
        self.losses = AuxiliaryLosses()

        #lingunet_params = self.root_params["LingUNet"]
        #self.encoder = Encoder(lingunet_params)
        self.encoder = WideEncoder()

        self.tripled_alpha = 1.0

        self.object_totals = {}
        self.object_mistake_counts = {}
        self.object_mistake_object_count = {}
        self.object_total_object_count = {}
        self.total_count = 0
        self.total_mistakes = 0

        self.runcount = 0

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

    def cam_poses_from_states(self, states):
        cam_pos = states[:, 9:12]
        cam_rot = states[:, 12:16]
        pose = Pose(cam_pos, cam_rot)
        return pose

    def maybe_cuda(self, tensor):
        return tensor.to(next(self.parameters()).device)

    def cuda_var(self, tensor):
        return tensor.to(next(self.parameters()).device)

    def unbatch(self, batch, halfway=False):
        obj_a, obj_b, obj_c, id_ab, id_c = batch
        obj_a = self.maybe_cuda(obj_a)
        obj_b = self.maybe_cuda(obj_b)
        obj_c = self.maybe_cuda(obj_c)
        return obj_a, obj_b, obj_c, id_ab, id_c

    def encode(self, img):
        x1, x2, x3, x4, x5 = self.encoder(img)
        vec = x5[:, :, 0, 0]
        return vec

    def score(self, vec_a, vec_b):
        #dist = ((vec_a - vec_b) ** 2).mean(1)
        dist = -((vec_a * vec_b).sum(1)) / (vec_a.norm(dim=1, p=2) * vec_b.norm(dim=1, p=2))
        return dist

    def forward(self, img_a, img_b):
        vec_a = self.encode(img_a)
        vec_b = self.encode(img_b)
        score = self.score(vec_a, vec_b)
        return score

    def reset_per_object_counts(self):
        self.object_totals = {}
        self.object_mistake_counts = {}
        self.object_mistake_object_count = {}
        self.object_total_object_count = {}
        self.total_count = 0
        self.total_mistakes = 0

    def per_object_counts(self, scores_ab, scores_ac, ids_ab, ids_c):
        for score_ab, score_ac, id_ab, id_c in zip(scores_ab, scores_ac, ids_ab, ids_c):
            correct = score_ab.item() < score_ac.item()
            self.total_count += 1
            if id_ab not in self.object_totals:
                self.object_totals[id_ab] = 0
            self.object_totals[id_ab] += 1
            if not correct:
                self.total_mistakes += 1
                if id_ab not in self.object_mistake_counts:
                    self.object_mistake_counts[id_ab] = 0
                self.object_mistake_counts[id_ab] += 1
                if id_ab not in self.object_mistake_object_count:
                    self.object_mistake_object_count[id_ab] = {}
                if id_c not in self.object_mistake_object_count[id_ab]:
                    self.object_mistake_object_count[id_ab][id_c] = 0
                self.object_mistake_object_count[id_ab][id_c] += 1
            if id_ab not in self.object_total_object_count:
                self.object_total_object_count[id_ab] = {}
            if id_c not in self.object_total_object_count[id_ab]:
                self.object_total_object_count[id_ab][id_c] = 0
            self.object_total_object_count[id_ab][id_c] += 1

    # Forward pass for training
    def sup_loss_on_batch(self, batch, eval, halfway=False, grad_noise=False, disable_losses=[]):
        self.prof.tick("out")

        if batch is None:
            print("Skipping None Batch")
            zero = torch.zeros([1]).float().to(next(self.parameters()).device)
            return zero, self.tensor_store

        obj_a, obj_b, obj_c, id_ab, id_c = self.unbatch(batch)
        self.prof.tick("unbatch_inputs")

        # ----------------------------------------------------------------------------
        vec_a = self.encode(obj_a)
        vec_b = self.encode(obj_b)
        vec_c = self.encode(obj_c)

        score_ab = self.score(vec_a, vec_b)
        score_ac = self.score(vec_a, vec_c)

        loss = torch.mean(F.relu(score_ab - score_ac + self.tripled_alpha))
        # ----------------------------------------------------------------------------
        # Tell the model prediction to the dataset, so that it can sample hard negatives
        (self.eval_dataset if eval else self.train_dataset).log_predictions(vec_a, vec_b, vec_c, id_ab, id_c)
        # ----------------------------------------------------------------------------
        if eval:
            self.runcount += 1
        if eval and self.runcount % 99 == 0:
            self.per_object_counts(score_ab, score_ac, id_ab, id_c)
            visualize_model(obj_a, obj_b, obj_c,
                            score_ab, score_ac,
                            self.total_count, self.total_mistakes,
                            self.object_mistake_counts, self.object_totals,
                            self.object_mistake_object_count, self.object_total_object_count,
                            self.get_iter(), self.run_name)
        else:
            self.reset_per_object_counts()
        # ----------------------------------------------------------------------------
        # The returned values are not used here - they're kept in the tensor store which is used as an input to a loss
        self.prof.tick("call")
        self.prof.tick("calc_losses")

        # Calculate metrics
        total = score_ab.shape[0] + score_ac.shape[0]
        correct_02 = ((score_ab < 0.2).sum().float() + (score_ac > 0.2).sum().float()) / (total + 1e-30)
        correct_05 = ((score_ab < 0.5).sum().float() + (score_ac > 0.5).sum().float()) / (total + 1e-30)
        correct_10 = ((score_ab < 1.0).sum().float() + (score_ac > 1.0).sum().float()) / (total + 1e-30)
        correct_15 = ((score_ab < 1.5).sum().float() + (score_ac > 1.5).sum().float()) / (total + 1e-30)
        correct_20 = ((score_ab < 2.0).sum().float() + (score_ac > 2.0).sum().float()) / (total + 1e-30)

        prefix = self.model_name + ("/eval" if eval else "/train")
        iteration = self.get_iter()
        self.writer.add_dict(prefix, get_current_meters(), iteration)
        self.writer.add_scalar(prefix, loss.item(), iteration)

        self.writer.add_scalar(prefix + "/accuracy_0.2", correct_02.item(), iteration)
        self.writer.add_scalar(prefix + "/accuracy_0.5", correct_05.item(), iteration)
        self.writer.add_scalar(prefix + "/accuracy_1.0", correct_10.item(), iteration)
        self.writer.add_scalar(prefix + "/accuracy_1.5", correct_15.item(), iteration)
        self.writer.add_scalar(prefix + "/accuracy_2.0", correct_20.item(), iteration)

        # Embedding distances so that we know roughly.
        mean_same_class_score = score_ab.mean().item()
        mean_diff_class_score = score_ac.mean().item()
        self.writer.add_scalar(prefix + "/same_class_dist", mean_same_class_score, iteration)
        self.writer.add_scalar(prefix + "/diff_class_dist", mean_diff_class_score, iteration)

        # Per-ID same-class scores
        for obj_id, score in zip(id_ab, score_ab):
            pass

        self.inc_iter()

        self.prof.tick("summaries")
        self.prof.loop()
        self.prof.print_stats(1)

        return loss, self.tensor_store

    def _init_dataset(self, eval=False):
        if eval:
            dataset_dir = P.get_current_parameters()["Data"].get("object_matching_test_dataset_dir")
        else:
            dataset_dir = P.get_current_parameters()["Data"].get("object_matching_training_dataset_dir")
        grayscale = P.get_current_parameters()["Data"].get("grayscale", False)
        hard_negative_mining = P.get_current_parameters()["Data"].get("hard_negative_mining", False)
        blur = P.get_current_parameters()["Data"].get("blur", False)
        noise = P.get_current_parameters()["Data"].get("noise", False)
        rotate = P.get_current_parameters()["Data"].get("rotate", False)
        flip = P.get_current_parameters()["Data"].get("flip", False)
        if grayscale:
            print("Matching network running GRAYSCALE!")
        return ObjectMatchingDataset(run_name=self.run_name, dataset_dir=dataset_dir, eval=eval,
                                     blur=blur, grain=noise, rotate=rotate, flip=flip,
                                     grayscale=grayscale, hard_negative_mining=hard_negative_mining)

    def get_dataset(self, eval=False):
        if eval and not self.eval_dataset:
            self.eval_dataset = self._init_dataset(eval)
        if not eval and not self.train_dataset:
            self.train_dataset = self._init_dataset(eval)
        return self.eval_dataset if eval else self.train_dataset
