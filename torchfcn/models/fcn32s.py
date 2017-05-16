import numpy as np
import torch
import torch.nn as nn


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class FCN32s(nn.Module):

    def __init__(self, n_class=21, nodeconv=False):
        super(FCN32s, self).__init__()
        self.nodeconv = nodeconv
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2

            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4

            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8

            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16

            # conv5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/32
        )
        self.classifier = nn.Sequential(
            # fc6
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            # fc7
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            # score_fr
            nn.Conv2d(4096, n_class, 1),
        )
        if self.nodeconv:
            self.upscore = nn.UpsamplingBilinear2d(scale_factor=32)
            self.upscore.scale_factor = None  # XXX: shoud be set in forward
        else:
            self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32,
                                              bias=False)

    def forward(self, x):
        h = self.features(x)

        h = self.classifier(h)

        if self.nodeconv:
            from chainer.utils import conv
            in_h, in_w = h.size()[2:4]
            out_h = conv.get_deconv_outsize(in_h, k=64, s=32, p=0)
            out_w = conv.get_deconv_outsize(in_w, k=64, s=32, p=0)
            self.upscore.size = out_h, out_w
            h = self.upscore(h)
        else:
            h = self.upscore(h)
        h = h[:, :, 19:19+x.size()[2], 19:19+x.size()[3]].contiguous()

        return h

    def copy_params_from_vgg16(self, vgg16, copy_fc8=True, init_upscore=True):
        for l1, l2 in zip(vgg16.features, self.features):
            if (isinstance(l1, nn.Conv2d) and
                    isinstance(l2, nn.Conv2d)):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i in [0, 3]:
            l1 = vgg16.classifier[i]
            l2 = self.classifier[i]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]
        if init_upscore:
            # initialize upscore layer
            c1, c2, h, w = self.upscore.weight.data.size()
            assert c1 == c2 == n_class
            assert h == w
            weight = get_upsample_filter(h)
            self.upscore.weight.data = \
                weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
