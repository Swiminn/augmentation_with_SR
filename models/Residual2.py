import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, num_channels, use_conv11=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=3, padding=1,
                               stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_conv11:
            self.conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=strides, padding=1)
        else:
            self.conv3 = None
        self.b1 = nn.BatchNorm2d(num_channels)
        self.b2 = nn.BatchNorm2d(num_channels)

    def forward(self, input):
        # print(input.shape)

        y = F.relu(self.b1(self.conv1(input)))
        y = self.b2(self.conv2(y))
        if self.conv3:
            input = self.conv3(input)
        # print(y.shape,input.shape)
        return F.relu(y + input)


def resnet18(num_classes, input_channels):
    net = nn.Sequential(
        nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(), )

    # print(net)

    def resnet_block(input_channel, num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add_module('Residual1', Residual(input_channel, num_channels, use_conv11=True, strides=2))
            else:
                blk.add_module('Residual', Residual(num_channels, num_channels, use_conv11=False))

        return blk

    net.add_module('block1', resnet_block(64, 64, 2, first_block=True))
    net.add_module('block2', resnet_block(64, 128, 2))
    net.add_module('block3', resnet_block(128, 256, 2))
    net.add_module('block4', resnet_block(256, 512, 2))
    net.add_module('pool', nn.AvgPool2d(3))
    net.add_module('dense1', nn.Conv2d(512, num_classes, kernel_size=1))
    # net.add_module('dense1', nn.Linear(120, num_classes))
    return net