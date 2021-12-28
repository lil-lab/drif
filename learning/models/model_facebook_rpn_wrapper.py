from detectron2.modeling.proposal_generator import rpn
from detectron2.structures.image_list import ImageList
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
from detectron2.layers.shape_spec import ShapeSpec
from detectron2.utils.events import EventStorage

# Uses Detectron2 version 3e994e353b0513c8994c3964b4585191f04ea14f

import torch
import torch.nn as nn
from data_io.paths import get_logging_dir
from learning.datasets.region_proposal_or_refinement_dataset import RegionProposalOrRefinementDataset
from learning.modules.key_tensor_store import KeyTensorStore
from learning.models.visualization.viz_html_facebook_rpn import visualize_model, draw_boxes
from learning.models.navigation_model_component_base import NavigationModelComponentBase

from utils.simple_profiler import SimpleProfiler
from utils.logging_summary_writer import LoggingSummaryWriter
from learning.meters_and_metrics.meter_server import get_current_meters
from utils.dummy_summary_writer import DummySummaryWriter
from utils.dict_tools import objectview

from visualization import Presenter

import parameters.parameter_server as P


# Mock FB config (they load this from YAML)
class MockFbConfig:
    class MockModelConfig:
        class MockAnchorGeneratorConfig:
            def __init__(self):
                self.NAME = "DefaultAnchorGenerator"
                self.SIZES = [[16, 32, 48, 64]]
                self.ASPECT_RATIOS = [[1.0]]
                self.ANGLES = [[0]]
                self.OFFSET = 0.5

        class MockRPNConfig:
            def __init__(self):
                self.HEAD_NAME = "StandardRPNHead"
                self.IN_FEATURES = [0]
                self.BOUNDARY_THRESH = -1
                self.IOU_THRESHOLDS = [0.3, 0.7]
                self.IOU_LABELS = [0, -1, 1]
                self.BATCH_SIZE_PER_IMAGE = 256
                self.POSITIVE_FRACTION = 0.5
                self.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
                self.SMOOTH_L1_BETA = 0.0
                self.LOSS_WEIGHT = 1.0
                #self.PRE_NMS_TOPK_TRAIN = 12000
                #self.PRE_NMS_TOPK_TEST = 6000
                #self.POST_NMS_TOPK_TRAIN = 2000
                #self.POST_NMS_TOPK_TEST = 1000
                self.PRE_NMS_TOPK_TRAIN = 500
                self.PRE_NMS_TOPK_TEST = 100
                self.POST_NMS_TOPK_TRAIN = 100
                self.POST_NMS_TOPK_TEST = 10
                self.NMS_THRESH = 0.7

        class MockProposalGeneratorConfig:
            def __init__(self):
                self.NAME = "RPN"
                self.MIN_SIZE = 0

        def __init__(self):
            self.RPN = self.MockRPNConfig()
            self.ANCHOR_GENERATOR = self.MockAnchorGeneratorConfig()
            self.PROPOSAL_GENERATOR = self.MockProposalGeneratorConfig()

    def __init__(self):
        self.MODEL = self.MockModelConfig()


class Encoder(torch.nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.p = objectview(params)

        # inchannels, outchannels, kernel size
        self.conv1 = nn.Conv2d(self.p.in_channels, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv2 = nn.Conv2d(self.p.hc1, self.p.hc1, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv4 = nn.Conv2d(self.p.hc1, self.p.hc1, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.norm2 = nn.InstanceNorm2d(self.p.hc1)
        self.norm3 = nn.InstanceNorm2d(self.p.hc1)
        self.norm4 = nn.InstanceNorm2d(self.p.hc1)
        self.norm5 = nn.InstanceNorm2d(self.p.hc1)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(p=0.5)

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
        #x3 = self.dropout(x3)
        x4 = self.act(self.conv4(x3))
        x5 = self.act(self.conv5(x4))
        return x1, x2, x3, x4, x5


class ModelFacebookRPNWrapper(NavigationModelComponentBase):

    def __init__(self, run_name="", domain="sim", nowriter=False):
        super(ModelFacebookRPNWrapper, self).__init__(run_name, domain, "region_proposal", nowriter)

        self.root_params = P.get_current_parameters()["ModelRegionProposalNetwork"]
        self.prof = SimpleProfiler(torch_sync=False, print=False)
        #self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

        lingunet_params = self.root_params["LingUNet"]
        self.encoder = Encoder(lingunet_params)

        self.fb_cfg = MockFbConfig()
        # TODO: Verify what format the shape should be
        shape = ShapeSpec(channels=32, width=16, height=12, stride=8)
        self.fb_rpn = rpn.RPN(cfg=self.fb_cfg, input_shape=[shape])

        self.runcount = 0
        self.tensor_store = KeyTensorStore()

    """
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

    """

    def init_weights(self):
        self.encoder.init_weights()

    def unbatch(self, batch, halfway=False):
        scene_images = self.to_model_device(batch["scene"])
        mask_images = self.to_model_device(batch["mask_union"])
        bboxes = [self.to_model_device(b) for b in batch["bboxes"]]
        return scene_images, mask_images, bboxes

    def encode(self, img):
        x1, x2, x3, x4, x5 = self.encoder(img)
        vec = x5
        return vec

    def forward(self, scene_images, bboxes=None):
        batch_size = scene_images.shape[0]
        # Wrap scene_images in the format FB wants
        image_shapes = [(scene_images.shape[2], scene_images.shape[3])] * batch_size
        scene_image_list = ImageList(scene_images, image_shapes)

        # Compute image features and wrap in the format FB wants:
        feature_map = self.encode(scene_images)
        features_dict = {0: feature_map}

        # Wrap bounding boxes, if any, in the format FB wants:
        if bboxes is not None:
            gt_instances = [Instances(image_shapes) for _ in range(batch_size)]
            for i in range(batch_size):
                gt_instances[i].gt_boxes = Boxes(bboxes[i])
        else:
            gt_instances = None

        with EventStorage():
            proposals, losses = self.fb_rpn(scene_image_list, features_dict, gt_instances)
        # proposals is a list of Instances, one for each image size (which I'll have only one for now)
        # losses is a dict of losses

        # Unpack bounding boxes from list of "Instances"
        instance = proposals[0]
        boxes_out = [instance.proposal_boxes.tensor for instance in proposals]
        logits_out = [instance.objectness_logits for instance in proposals]
        objectness_probs_out = [torch.sigmoid(l) for l in logits_out]

        # Filter by logits:
        PROB_THRESHOLD = 0.4
        filter_indices = [[i for i, prob in enumerate(ex_probs) if prob > PROB_THRESHOLD] for ex_probs in objectness_probs_out]
        filtered_boxes_out = [[ex_bboxes[i] for i in ex_indices] for ex_bboxes, ex_indices in zip(boxes_out, filter_indices)]
        filtered_probs_out = [[ex_probs[i] for i in ex_indices] for ex_probs, ex_indices in zip(objectness_probs_out, filter_indices)]

        # TODO: Perhaps add a logit-based filtering step
        if bboxes is not None:
            return filtered_boxes_out, filtered_probs_out, losses
        else:
            return filtered_boxes_out, filtered_probs_out

    # Forward pass for training
    def sup_loss_on_batch(self, batch, eval, halfway=False, grad_noise=False, disable_losses=[]):
        self.prof.tick("out")
        if batch is None:
            print("Skipping None Batch")
            zero = torch.zeros([1]).float().to(next(self.parameters()).device)
            return zero, self.tensor_store

        scene_images, mask_images, gt_bboxes = self.unbatch(batch)
        self.prof.tick("unbatch_inputs")

        # ----------------------------------------------------------------------------
        proposal_boxes, probs, losses = self(scene_images, gt_bboxes)
        # ----------------------------------------------------------------------------
        # Visualize the model
        if eval:
            self.runcount += 1
            if self.runcount % 100 == 0:
                visualize_model(scene_images, proposal_boxes, self.get_iter(), self.run_name)
        if self.get_iter() % 3 == 0:
            scene_image = scene_images[0].detach().cpu()
            bboxes = proposal_boxes[0]
            scene_image_with_boxes = draw_boxes(scene_image, bboxes)
            Presenter().show_image(scene_image_with_boxes, "proposals", scale=4, waitkey=1)
            scene_image_with_ground_truth_boxes = draw_boxes(scene_image, gt_bboxes[0])
            Presenter().show_image(scene_image_with_ground_truth_boxes, "ground_truth", scale=4, waitkey=1)

        # ----------------------------------------------------------------------------
        # The returned values are not used here - they're kept in the tensor store which is used as an input to a loss
        self.prof.tick("call")
        self.prof.tick("calc_losses")

        loss = sum([v for k, v in losses.items()])

        prefix = self.model_name + ("/eval" if eval else "/train")
        iteration = self.get_iter()
        self.writer.add_dict(prefix, get_current_meters(), iteration)
        self.writer.add_scalar(prefix, loss if isinstance(loss, int) else loss.item(), iteration)

        self.inc_iter()

        self.prof.tick("summaries")
        self.prof.loop()
        self.prof.print_stats(1)

        return loss, self.tensor_store

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
            mode="region_proposal",
            eval=eval,
            blur=blur, grain=noise, flip=flip, grayscale=grayscale)

    def get_dataset(self, eval=False):
        return self._init_dataset(eval)
