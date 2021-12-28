import torch
import spacy

from torch import nn
from learning.modules.grounding.metric_similarity_base import MetricSimilarityBase
import torchvision.models as models

from learning.modules.unet.lingunet_5_instance_det import Lingunet5Encoder
from learning.modules.instance_detector.feature_extractor import FeatureExtractor
from utils.dict_tools import objectview

import parameters.parameter_server as P

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
        x1 = self.act(self.conv1(img))
        if img.shape[0] > 0:
            x1 = self.norm2(x1)
        x2 = self.act(self.conv2(x1))
        if img.shape[0] > 0:
            x2 = self.norm3(x2)
        x3 = self.act(self.conv3(x2))
        #x3 = self.dropout(x3)
        x4 = self.act(self.conv4(x3))
        x5 = self.act(self.conv5(x4))
        return x1, x2, x3, x4, x5


class ImageMetricEmbedding(nn.Module):
    # TODO: Implement
    def __init__(self):
        super().__init__()
        param_dict = P.get_current_parameters()["ImageEmbedding"]
        self.encoder = Encoder(param_dict)

    def init_weights(self):
        self.encoder.init_weights()

    def encode(self, img):
        x1, x2, x3, x4, x5 = self.encoder(img)
        vec = x5[:, :, 0, 0]
        return vec

    def batch_encode(self, imgs):
        bs = imgs.shape[0]
        q = imgs.shape[1]
        c = imgs.shape[2]
        h = imgs.shape[3]
        w = imgs.shape[4]
        imgs = imgs.view([bs * q, c, h, w])
        vecs = self.encode(imgs)
        vdim = vecs.shape[1]
        vecs = vecs.view([bs, q, vdim])
        return vecs
