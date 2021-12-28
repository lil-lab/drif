import torch.nn as nn
from learning.modules.unet.lingunet_5 import Lingunet5
from learning.modules.unet.lingunet_5_oob import Lingunet5OOB
from learning.modules.spatial_softmax_2d import SpatialSoftmax2d
from copy import deepcopy

sentence_embedding_size = 120
sentence_embedding_layers = 1
word_embedding_size = 20


class LingunetPathPredictor(nn.Module):

    def __init__(self, lingunet_params,
                 posterior_channels_in,
                 oob=False,
                 noling=False):
        super(LingunetPathPredictor, self).__init__()

        self.posterior_img_channels = posterior_channels_in
        self.small_network = lingunet_params.get("small_network")
        self.oob = oob
        self.noling = noling

        lingunet_params["noling"] = self.noling
        if self.noling:
            lingunet_params["hb1"] = lingunet_params["hc1"]

        if self.oob:
            lingunet_params["in_channels"] = posterior_channels_in
            self.unet_posterior = Lingunet5OOB(deepcopy(lingunet_params))
        else:
            lingunet_params["in_channels"] = posterior_channels_in
            self.unet_posterior = Lingunet5(deepcopy(lingunet_params))

        self.softmax = SpatialSoftmax2d()
        self.norm = nn.InstanceNorm2d(2)

        self.dbg_t = None

    def init_weights(self):
        self.unet_posterior.init_weights()

    def forward(self, image, sentence_embeddimg, map_poses, tensor_store=None, show=""):

        # TODO: Move map perturb data augmentation in here.
        if image.size(1) > self.posterior_img_channels:
            image = image[:, 0:self.posterior_img_channels, :, :]

        # The first N channels would've been computed by grounding map processor first. Remove them so that the
        # prior is clean from any language
        posterior_distributions = self.unet_posterior(image, sentence_embeddimg, tensor_store)

        return posterior_distributions, map_poses
