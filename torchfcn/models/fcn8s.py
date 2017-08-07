import numpy as np
import torch
import torch.nn as nn


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN8s(nn.Module):
    def __init__(self, n_class=21):
        super(FCN8s, self).__init__()

        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
        )

        # conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
        )

        # conv3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
        )

        # conv4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
        )

        # conv5
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/32
        )

        self.features = nn.Sequential(
            self.conv1, self.conv2, self.conv3, self.conv4, self.conv5
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

        self.upscore2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2,
                                           bias=False)
        self.upscore4 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2,
                                           bias=False)
        self.upscore8 = nn.ConvTranspose2d(n_class, n_class, 16, stride=8,
                                           bias=False)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

    def forward(self, x):
        conv3 = self.conv3(self.conv2(self.conv1(x)))
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        score_fr = self.classifier(conv5)

        score_pool3 = self.score_pool3(conv3)
        score_pool4 = self.score_pool4(conv4)

        # upscore2, crop score_pool4 and get fuse_pool4
        upscore2 = self.upscore2(score_fr)
        score_pool4_crop = score_pool4[:, :, 5: 5 + upscore2.size()[2],
                                       5: 5 + upscore2.size()[3]]
        fuse_pool4 = upscore2 + score_pool4_crop

        # upscore4, crop upscore_fuse_pool4 and get fuse_pool3
        upscore4 = self.upscore4(fuse_pool4)
        score_pool3_crop = score_pool3[:, :, 9: 9 + upscore4.size()[2],
                                       9: 9 + upscore4.size()[3]]
        fuse_pool3 = upscore4 + score_pool3_crop

        # upscore8 and crop it
        upscore8 = self.upscore8(fuse_pool3)
        score = upscore8[:, :, 31: 31 + x.size()[2],
                         31: 31 + x.size()[3]].contiguous()

        return score

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTransposed2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def copy_params_from_vgg16(self, vgg16):
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
