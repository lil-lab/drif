import torch
from torch.autograd import Variable
from torch import nn as nn
import torch.nn.functional as F
from torch.nn.modules.upsampling import Upsample

from learning.inputs.partial_2d_distribution import Partial2DDistribution

from utils.dict_tools import objectview


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


class DoubleDeconv(torch.nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super(DoubleDeconv, self).__init__()
        self.conv1 = nn.ConvTranspose2d(cin, cout, k, stride=1, padding=padding)
        self.conv2 = nn.ConvTranspose2d(cout, cout, k, stride=stride, padding=padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, img, output_size):
        # TODO: 2 is stride
        osize1 = [int(i/2) for i in output_size]
        osize1[2] = max(osize1[2], img.shape[2])
        osize1[3] = max(osize1[3], img.shape[3])
        x = self.conv1(img, output_size=osize1)
        x = F.leaky_relu(x)
        x = self.conv2(x, output_size=output_size)
        return x


class UpscaleDoubleConv(torch.nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super(UpscaleDoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(cin, cout, k, stride=1, padding=padding)
        #self.upsample1 = Upsample(scale_factor=2, mode="nearest")
        self.conv2 = nn.Conv2d(cout, cout, k, stride=1, padding=padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, img, output_size):
        x = self.conv1(img)
        x = F.leaky_relu(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        #x = self.upsample1(x)
        x = self.conv2(x)
        return x


class UpscaleConv(torch.nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super(UpscaleConv, self).__init__()
        self.upsample1 = Upsample(scale_factor=2, mode="nearest")
        self.conv2 = nn.Conv2d(cin, cout, k, stride=1, padding=padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, img, output_size):
        x = F.leaky_relu(img)
        x = self.upsample1(x)
        x = self.conv2(x)
        return x


class Lingunet5Encoder(torch.nn.Module):
    def __init__(self, params):
        super(Lingunet5Encoder, self).__init__()
        self.p = objectview(params)

        # inchannels, outchannels, kernel size
        self.conv1 = DoubleConv(self.p.in_channels, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv2 = DoubleConv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv3 = DoubleConv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv4 = DoubleConv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv5 = DoubleConv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.norm2 = nn.InstanceNorm2d(self.p.hc1)
        self.norm3 = nn.InstanceNorm2d(self.p.hc1)
        self.norm4 = nn.InstanceNorm2d(self.p.hc1)
        self.norm5 = nn.InstanceNorm2d(self.p.hc1)
        self.act = nn.LeakyReLU()

    def init_weights(self):
        self.conv1.init_weights()
        self.conv2.init_weights()
        self.conv3.init_weights()
        self.conv4.init_weights()
        self.conv5.init_weights()

    def forward(self, img):
        x1 = self.norm2(self.act(self.conv1(img)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))
        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.act(self.conv5(x4))
        return x1, x2, x3, x4, x5


class Lingunet5Filter(torch.nn.Module):
    def __init__(self, params):
        super(Lingunet5Filter, self).__init__()

        self.p = objectview(params)
        self.fnorm1 = nn.InstanceNorm2d(self.p.filter_out_c)
        self.fnorm2 = nn.InstanceNorm2d(self.p.filter_out_c)
        self.fnorm3 = nn.InstanceNorm2d(self.p.filter_out_c)
        self.fnorm4 = nn.InstanceNorm2d(self.p.filter_out_c)
        self.act = nn.LeakyReLU()

    def forward(self, x1, x2, x3, x4, x5, l1fs, l2fs, l3fs, l4fs, l5fs):
        batch_size = x1.shape[0]
        # These conv filters are different for each element in the batch, but the functional convolution
        # operator assumes the same filters across the batch.
        # TODO: Verify if slicing like this is a terrible idea for performance
        x1f = torch.zeros([x1.shape[0], self.p.filter_out_c, x1.shape[2], x1.shape[3]], device=x1.device)
        x2f = torch.zeros([x2.shape[0], self.p.filter_out_c, x2.shape[2], x2.shape[3]], device=x2.device)
        x3f = torch.zeros([x3.shape[0], self.p.filter_out_c, x3.shape[2], x3.shape[3]], device=x3.device)
        x4f = torch.zeros([x4.shape[0], self.p.filter_out_c, x4.shape[2], x4.shape[3]], device=x4.device)
        x5f = torch.zeros([x5.shape[0], self.p.filter_out_c, x5.shape[2], x5.shape[3]], device=x5.device)

        for i in range(batch_size):
            x1f[i:i + 1] = F.conv2d(x1[i:i + 1], l1fs[i])
            x2f[i:i + 1] = F.conv2d(x2[i:i + 1], l2fs[i])
            x3f[i:i + 1] = F.conv2d(x3[i:i + 1], l3fs[i])
            x4f[i:i + 1] = F.conv2d(x4[i:i + 1], l4fs[i])
            x5f[i:i + 1] = F.conv2d(x5[i:i + 1], l5fs[i])

        x1 = self.fnorm1(x1f)
        x2 = self.fnorm2(x2f)
        x3 = self.fnorm3(x3f)
        x4 = self.fnorm4(x4f)
        x5 = x5f
        return x1, x2, x3, x4, x5


class Lingunet5GroupFilter(torch.nn.Module):
    """
    Filters a sequence of intermediate features using grouped convolutions.
    """
    def __init__(self, params):
        super(Lingunet5GroupFilter, self).__init__()

        self.p = objectview(params)
        self.fnorm1 = nn.InstanceNorm2d(self.p.hb1)
        self.fnorm2 = nn.InstanceNorm2d(self.p.hb1)
        self.fnorm3 = nn.InstanceNorm2d(self.p.hb1)
        self.fnorm4 = nn.InstanceNorm2d(self.p.hb1)
        self.act = nn.LeakyReLU()

    def layer_forward(self, x, lf):
        batch_size = x.shape[0]
        x_out = torch.zeros([x.shape[0], self.out_channels, x.shape[2], x.shape[3]], device=x.device)
        for i in range(batch_size):
            x_out[i:i+1] = F.conv2d()

    def forward(self, x1, x2, x3, x4, x5, l1fs, l2fs, l3fs, l4fs, l5fs):
        """
        :param x1-5: BxCxHxW feature maps
        :param l1-5fs: BxGxOxCx1x1 kernels
        :return: BxOxCx1x1 feature maps
          ... each output feature map in the batch is obtained by convolving with each of the G kernels for that
          batch element, and then max-pooling along the G-dimension.
        """
        batch_size = x1.shape[0]
        num_groups = len(l1fs)
        # These conv filters are different for each element in the batch, but the functional convolution
        # operator assumes the same filters across the batch.
        # TODO: Verify if slicing like this is a terrible idea for performance
        x1f = torch.zeros([x1.shape[0], self.p.hb1, x1.shape[2], x1.shape[3]], device=x1.device)
        x2f = torch.zeros([x2.shape[0], self.p.hb1, x2.shape[2], x2.shape[3]], device=x2.device)
        x3f = torch.zeros([x3.shape[0], self.p.hb1, x3.shape[2], x3.shape[3]], device=x3.device)
        x4f = torch.zeros([x4.shape[0], self.p.hb1, x4.shape[2], x4.shape[3]], device=x4.device)
        x5f = torch.zeros([x5.shape[0], self.p.hb1, x5.shape[2], x5.shape[3]], device=x5.device)

        for i in range(batch_size):
            x1f[i:i + 1] = F.conv2d(x1[i:i + 1], l1fs[i])
            x2f[i:i + 1] = F.conv2d(x2[i:i + 1], l2fs[i])
            x3f[i:i + 1] = F.conv2d(x3[i:i + 1], l3fs[i])
            x4f[i:i + 1] = F.conv2d(x4[i:i + 1], l4fs[i])
            x5f[i:i + 1] = F.conv2d(x5[i:i + 1], l5fs[i])

        x1 = self.fnorm1(x1f)
        x2 = self.fnorm2(x2f)
        x3 = self.fnorm3(x3f)
        x4 = self.fnorm4(x4f)
        x5 = x5f
        return x1, x2, x3, x4, x5


class Lingunet5FilterMaker(torch.nn.Module):
    def __init__(self, params):
        super(Lingunet5Filter, self).__init__()

        self.p = objectview(params)
        self.lang19 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang28 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang37 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang46 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang55 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hc1)
        self.fnorm1 = nn.InstanceNorm2d(self.p.hb1)
        self.fnorm2 = nn.InstanceNorm2d(self.p.hb1)
        self.fnorm3 = nn.InstanceNorm2d(self.p.hb1)
        self.fnorm4 = nn.InstanceNorm2d(self.p.hb1)
        self.act = nn.LeakyReLU()

    def forward(self, x1, x2, x3, x4, x5, embedding):
        batch_size = x1.shape[0]
        embedding = F.normalize(embedding, p=2, dim=1)
        if self.p.split_embedding:
            block_size = self.emb_block_size
            emb1 = embedding[:, 0 * block_size:1 * block_size]
            emb2 = embedding[:, 1 * block_size:2 * block_size]
            emb3 = embedding[:, 2 * block_size:3 * block_size]
            emb4 = embedding[:, 3 * block_size:4 * block_size]
            emb5 = embedding[:, 4 * block_size:5 * block_size]
        else:
            emb1 = emb2 = emb3 = emb4 = emb5 = embedding

        l1fs = []
        l2fs = []
        l3fs = []
        l4fs = []
        l5fs = []
        for i in range(batch_size):
            emb_idx = i if embedding.shape[0] == batch_size else 0
            lf1 = F.normalize(self.lang19(emb1[emb_idx:emb_idx + 1])).view([self.p.hb1, self.p.hc1, 1, 1])
            lf2 = F.normalize(self.lang28(emb2[emb_idx:emb_idx + 1])).view([self.p.hb1, self.p.hc1, 1, 1])
            lf3 = F.normalize(self.lang37(emb3[emb_idx:emb_idx + 1])).view([self.p.hb1, self.p.hc1, 1, 1])
            lf4 = F.normalize(self.lang46(emb4[emb_idx:emb_idx + 1])).view([self.p.hb1, self.p.hc1, 1, 1])
            lf5 = F.normalize(self.lang55(emb5[emb_idx:emb_idx + 1])).view([self.p.hc1, self.p.hc1, 1, 1])
            l1fs.append(lf1)
            l2fs.append(lf2)
            l3fs.append(lf3)
            l4fs.append(lf4)
            l5fs.append(lf5)
        return l1fs, l2fs, l3fs, l4fs, l5fs


class Lingunet5Decoder(torch.nn.Module):
    def __init__(self, params):
        super(Lingunet5Decoder, self).__init__()
        self.p = objectview(params)
        if self.p.upscale_conv:
            if self.p.double_up:
                DeconvOp = UpscaleDoubleConv
            else:
                DeconvOp = UpscaleConv
            LastDeconvOp = DeconvOp
        else:
            DeconvOp = DoubleDeconv
            LastDeconvOp = nn.ConvTranspose2d

        self.deconv1 = DeconvOp(self.p.hb1, self.p.hc2, 3, stride=self.p.stride, padding=1)
        self.deconv2 = DeconvOp(self.p.hc2 + self.p.hb1, self.p.hc2, 3, stride=self.p.stride, padding=1)
        self.deconv3 = DeconvOp(self.p.hc2 + self.p.hb1, self.p.hc2, 3, stride=self.p.stride, padding=1)
        self.deconv4 = DeconvOp(self.p.hc2 + self.p.hb1, self.p.hc3, 3, stride=self.p.stride, padding=1)
        self.deconv5 = LastDeconvOp(self.p.hb1 + self.p.hc3, self.p.out_channels, 3, stride=self.p.stride, padding=1)

        # self.dnorm1 = nn.InstanceNorm2d(in_channels * 4)
        self.dnorm2 = nn.InstanceNorm2d(self.p.hc2)
        self.dnorm3 = nn.InstanceNorm2d(self.p.hc2)
        self.dnorm4 = nn.InstanceNorm2d(self.p.hc2)
        self.dnorm5 = nn.InstanceNorm2d(self.p.hc3)
        self.act = nn.LeakyReLU()

    def init_weights(self):
        self.deconv1.init_weights()
        self.deconv2.init_weights()
        self.deconv3.init_weights()
        self.deconv4.init_weights()
        #self.deconv5.init_weights()

    def forward(self, input, x1, x2, x3, x4, x5):
        x6 = self.act(self.deconv1(x5, output_size=x4.size()))
        x46 = torch.cat([x4, x6], 1)
        x7 = self.dnorm3(self.act(self.deconv2(x46, output_size=x3.size())))
        x37 = torch.cat([x3, x7], 1)
        x8 = self.dnorm4(self.act(self.deconv3(x37, output_size=x2.size())))
        x28 = torch.cat([x2, x8], 1)
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        x19 = torch.cat([x1, x9], 1)
        xout = self.deconv5(x19, output_size=input.size())
        return x46, x37, x28, x19, xout


class Lingunet5InstanceDet(torch.nn.Module):
    def __init__(self, params):
                 #in_channels, out_channels, embedding_size, hc1=32, hb1=16, hc2=256, stride=2, split_embedding=False):
        super(Lingunet5InstanceDet, self).__init__()

        self.p = objectview(params)

        if self.p.split_embedding:
            self.emb_block_size = int(self.p.embedding_size / 5)
        else:
            self.emb_block_size = self.p.embedding_size

        self.encoder = Lingunet5Encoder(params)
        self.filter_maker = Lingunet5FilterMaker(params)
        self.conditional_filter = Lingunet5Filter(params)
        self.decoder = Lingunet5Decoder(params)

        self.convoob = DoubleConv(self.p.hb1 + self.p.hc2, self.p.out_channels, 3, stride=2, padding=1, stride2=2, cmid=16)
        self.act = nn.LeakyReLU()

    def init_weights(self):
        self.encoder.init_weights()
        self.decoder.init_weights()

    def forward(self, input, embedding, tensor_store=None):
        batch_size = input.shape[0]

        x1, x2, x3, x4, x5 = self.encoder(input)

        if tensor_store is not None:
            tensor_store.keep_inputs("lingunet_f1", x1)
            tensor_store.keep_inputs("lingunet_f2", x2)
            tensor_store.keep_inputs("lingunet_f3", x3)
            tensor_store.keep_inputs("lingunet_f4", x4)
            tensor_store.keep_inputs("lingunet_f5", x5)

        l1fs, l2fs, l3fs, l4fs, l5fs = self.filter_maker(x1, x2, x3, x4, x5, embedding)
        x1, x2, x3, x4, x5 = self.conditional_filter(x1, x2, x3, x4, x5, l1fs, l2fs, l3fs, l4fs, l5fs)

        if tensor_store is not None:
            tensor_store.keep_inputs("lingunet_g1", x1)
            tensor_store.keep_inputs("lingunet_g2", x2)
            tensor_store.keep_inputs("lingunet_g3", x3)
            tensor_store.keep_inputs("lingunet_g4", x4)
            tensor_store.keep_inputs("lingunet_g5", x5)

        x46, x37, x28, x19, xout = self.decoder(input, x1, x2, x3, x4, x5)
        # Predict probability masses / scores for the goal or trajectory traveling outside the observed part of the map
        o = self.convoob(x19)
        outer_scores = F.avg_pool2d(o, o.shape[2]).view([batch_size, self.p.out_channels])
        both_dist_scores = Partial2DDistribution(xout, outer_scores)
        return both_dist_scores
