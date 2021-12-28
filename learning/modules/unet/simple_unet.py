import torch
from torch import nn as nn
import torch.nn.functional as F

from utils.dict_tools import objectview


class Conv(torch.nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(cin, cout, k, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(cout, cout, k, stride=1, padding=padding)

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
        self.conv1 = nn.ConvTranspose2d(cin, cout, k, stride=stride, padding=padding)
        self.conv2 = nn.ConvTranspose2d(cout, cout, k, stride=1, padding=padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, img, output_size):
        # TODO: 2 is stride
        osize1 = [int(i/2) for i in output_size]
        x = self.conv1(img, output_size=output_size)
        x = F.leaky_relu(x)
        x = self.conv2(x, output_size=output_size)
        return x


class SimleUNet(torch.nn.Module):
    def __init__(self, params):
        super(SimleUNet, self).__init__()

        self.p = objectview(params)

        # inchannels, outchannels, kernel size
        self.conv1 = Conv(self.p.in_channels, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv2 = Conv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv3 = Conv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv4 = Conv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv5 = Conv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)

        self.deconv1 = DoubleDeconv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv2 = DoubleDeconv(self.p.hc1 + self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv3 = DoubleDeconv(self.p.hc1 + self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv4 = DoubleDeconv(self.p.hc1 + self.p.hc1, self.p.hc2, 3, stride=self.p.stride, padding=1)
        self.deconv5 = nn.ConvTranspose2d(self.p.hc1 + self.p.hc2, self.p.hc3, 3, stride=self.p.stride, padding=1)
        self.deconv5_out = nn.ConvTranspose2d(self.p.hc3, self.p.out_channels, 3, stride=1, padding=1)

        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.norm2 = nn.InstanceNorm2d(self.p.hc1)
        self.norm3 = nn.InstanceNorm2d(self.p.hc1)
        self.norm4 = nn.InstanceNorm2d(self.p.hc1)
        self.norm5 = nn.InstanceNorm2d(self.p.hc1)
        # self.dnorm1 = nn.InstanceNorm2d(in_channels * 4)
        self.dnorm2 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm3 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm4 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm5 = nn.InstanceNorm2d(self.p.hc2)

    def init_weights(self):
        self.conv1.init_weights()
        self.conv2.init_weights()
        self.conv3.init_weights()
        self.conv4.init_weights()
        self.conv5.init_weights()
        self.deconv1.init_weights()
        self.deconv2.init_weights()
        self.deconv3.init_weights()
        self.deconv4.init_weights()

    def forward(self, input):
        x1 = self.norm2(self.act(self.conv1(input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))
        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.act(self.conv5(x4))

        x6 = self.act(self.deconv1(x5, output_size=x4.size()))
        x46 = torch.cat([x4, x6], 1)
        x7 = self.dnorm3(self.act(self.deconv2(x46, output_size=x3.size())))
        x37 = torch.cat([x3, x7], 1)
        x8 = self.dnorm4(self.act(self.deconv3(x37, output_size=x2.size())))
        x28 = torch.cat([x2, x8], 1)
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        x19 = torch.cat([x1, x9], 1)
        xout = self.deconv5(x19, output_size=input.size())
        out = self.deconv5_out(xout, output_size=input.size())

        return out
