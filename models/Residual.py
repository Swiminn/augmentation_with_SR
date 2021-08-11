import torch
from torch import nn
from torch.nn import functional as F


'''
ResBlock
'''
class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out, stride=1):
        # Reduce parameter dimensions through Strie
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)


        # [b, ch_in, h, w] => [b, ch_out, h, w]
        self.extra = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        '''
        :param x: [b, ch, h, w]
        :return:
        '''
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out

        return out


'''
ResNet-18
'''
class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16)
        )

        # follow 4 blocks
        # [b, 16, h, w] => [b, 16, h, w]
        self.blk1 = ResBlk(16, 16, stride=1)
        self.blk2 = ResBlk(16, 16, stride=1)
        self.blk3 = ResBlk(16, 16, stride=1)
        # [b, 16, h, w] => [b, 32, h/2, w/2]
        self.blk4 = ResBlk(16, 32, stride=2)
        self.blk5 = ResBlk(32, 32, stride=1)
        self.blk6 = ResBlk(32, 32, stride=1)
        # [b, 32, h/2, w/2] => [b, 64, h/4, w/4]
        self.blk7 = ResBlk(32, 64, stride=2)
        self.blk8 = ResBlk(64, 64, stride=1)
        self.blk9 = ResBlk(64, 64, stride=1)
        self.out_layer = nn.Linear(64*1*1, 10)

    def forward(self, x):

        # [b, 3, h, w] => [b, 16, h, w]
        x = F.relu(self.conv1(x))

        # [b, 16, h, w] => [b, 64, h/4, w/4]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = self.blk5(x)
        x = self.blk6(x)
        x = self.blk7(x)
        x = self.blk8(x)

        # [b, 64, h/4, w/4] => [b, 64, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])

        # [b, 64, 1, 1] => [b, 64]
        x = x.view(x.size(0), -1)
        # [b, 64] => [b, 10]
        x = self.out_layer(x)

        return x


if __name__ == '__main__':
    res_net = ResNet18()
    tmp = torch.randn(2,3,32,32)
    print(res_net.forward(tmp).shape)