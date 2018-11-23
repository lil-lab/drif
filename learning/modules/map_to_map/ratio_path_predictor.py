import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from data_io.weights import enable_weight_saving
from learning.inputs.common import empty_float_tensor, cuda_var
from learning.modules.map_transformer_base import MapTransformerBase
#from learning.modules.unet.unet_5_contextual_bneck import Unet5ContextualBneck
from learning.modules.unet.unet_5_contextual_bneck3 import Unet5ContextualBneck
from learning.modules.rss.map_lang_semantic_filter import MapLangSemanticFilter
from learning.modules.spatial_softmax_2d import SpatialSoftmax2d

from parameters.parameter_server import get_current_parameters
from learning.modules.dbg_writer import DebugWriter
from visualization import Presenter

sentence_embedding_size = 120
sentence_embedding_layers = 1
word_embedding_size = 20


# TODO: Rename to ImgToImg
# TODO: Remove the transforming bits - that should be handled by MapTransformer
class RatioPathPredictor(MapTransformerBase):

    # TODO: Standardize run_params
    def __init__(self, prior_channels=32,
                 posterior_channels=32,
                 pred_channels=2,
                 emb_size=120,
                 source_map_size=32,
                 world_size=32,
                 compute_prior=True,
                 use_prior=False,
                 l2=False):
        super(RatioPathPredictor, self).__init__(source_map_size, world_size)

        self.prior_img_channels = prior_channels
        self.posterior_img_channels = posterior_channels
        self.emb_size = emb_size
        self.l2 = l2
        self.use_prior = use_prior

        if use_prior:
            assert compute_prior, "If we want to use the prior distribution, we should compute it, right?"

        self.unet_posterior = Unet5ContextualBneck(
            posterior_channels,
            pred_channels,
            emb_size,
            hc1=48, hb1=24, hc2=128)

        self.unet_prior = Unet5ContextualBneck(
            prior_channels,
            pred_channels,
            1,
            hc1=48, hb1=24, hc2=128)

        self.softmax = SpatialSoftmax2d()
        self.norm = nn.InstanceNorm2d(2)
        self.compute_prior = compute_prior

        #self.map_filter = MapLangSemanticFilter(emb_size, feature_channels, 3)
        self.map_size = source_map_size
        self.world_size = world_size

        self.dbg_t = None
        self.seq = 0

    def init_weights(self):
        self.unet_posterior.init_weights()
        self.unet_prior.init_weights()

    def reset(self):
        super(RatioPathPredictor, self).reset()
        self.seq = 0

    def cuda(self, device=None):
        MapTransformerBase.cuda(self, device)
        #self.map_filter.cuda(device)
        self.softmax.cuda(device)
        self.dbg_t = None
        return self

    def dbg_write_extra(self, map, pose):
        if DebugWriter().should_write():
            self.seq += 1
            # Initialize a transformer module
            if self.dbg_t is None:
                self.dbg_t = MapTransformerBase(self.map_size, self.world_size)
                if self.is_cuda:
                    self.dbg_t.cuda(self.cuda_device)

            # Transform the prediction to the global frame and write out to disk.
            self.dbg_t.set_map(map, pose)
            map_global, _ = self.dbg_t.get_map(None)
            DebugWriter().write_img(map_global[0], "gif_overlaid", args={"world_size": self.world_size, "name": "pathpred"})

    def forward(self, image, sentence_embeddimg, map_poses, value_store=None, show=""):

        # TODO: Move map perturb data augmentation in here.
        if image.size(1) > self.posterior_img_channels:
            image = image[:, 0:self.posterior_img_channels, :, :]

        # channel 0 is start position
        # channels 1-3 are the grounded map
        # all other channels are the semantic map
        fake_embedding = Variable(empty_float_tensor([image.size(0), 1], self.is_cuda, self.cuda_device))

        # The first N channels would've been computed by grounding map processor first. Remove them so that the
        # prior is clean from any language

        pred_mask_posterior = self.unet_posterior(image, sentence_embeddimg)
        if not self.l2:
            pred_mask_posterior_prob = self.softmax(pred_mask_posterior)
        else:
            pred_mask_posterior_prob = pred_mask_posterior

        if self.compute_prior:
            lang_conditioned_channels = self.posterior_img_channels - self.prior_img_channels
            prior_image = image[:, lang_conditioned_channels:]
            pred_mask_prior = self.unet_prior(prior_image, fake_embedding)
            if not self.l2:
                pred_mask_prior_prob = self.softmax(pred_mask_prior)
            else:
                pred_mask_prior_prob = pred_mask_prior
            ratio_mask = pred_mask_posterior_prob / (pred_mask_prior_prob + 1e-3)
            ratio_mask = self.softmax(ratio_mask)
        else:
            pred_mask_prior = pred_mask_posterior

        #if show != "":
        #    Presenter().show_image(ratio_mask.data[i], show, torch=True, scale=8, waitkey=1)

        self.set_maps(pred_mask_posterior_prob, map_poses)

        ret = pred_mask_posterior_prob
        if self.use_prior:
            ret = pred_mask_prior_prob

        return ret, pred_mask_prior, pred_mask_posterior, map_poses