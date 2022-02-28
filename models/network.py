import torch.nn as nn
from core.base_network import BaseNetwork
class Network(BaseNetwork):
    def __init__(self, in_channels=3, **kwargs):
        super(Network, self).__init__(**kwargs)

        self.in_channels = in_channels
        cnums = 64
        self.down_net = nn.Sequential(
            Down2(in_channels, cnums),
            Down2(cnums, cnums*2),
            Down3(cnums*2, cnums*4),
            Down2(cnums*4, cnums*8),
        )
        self.up_net = nn.Sequential(
            Up2(cnums*8, cnums*4),
            Up2(cnums*4, cnums*2),
            Up3(cnums*2, cnums*1),
            # Up2(cnums*1, cnums*1),
            nn.Upsample(scale_factor=2, mode='nearest', align_corners=None),
            nn.Conv2d(cnums*1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, _x):
        _x = self.down_net(_x)
        _x = self.up_net(_x)
        return _x

        
class conv2DBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            with_bn=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation, )

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.ReLU(inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class Down2(nn.Module):
    def __init__(self, in_size, out_size):
        super(Down2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 2, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class Down3(nn.Module):
    def __init__(self, in_size, out_size):
        super(Down3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 2, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class Up2(nn.Module):
    def __init__(self, in_size, out_size):
        super(Up2, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs):
        outputs = self.up(inputs)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class Up3(nn.Module):
    def __init__(self, in_size, out_size):
        super(Up3, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs):
        outputs = self.up(inputs)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs

