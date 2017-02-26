import torch

from torchfcn.utils import conv


def copy_params_vgg16_to_fcn32s(vgg16, fcn32s,
                                copy_fc8=True, init_upscore=True):
    for l1, l2 in zip(vgg16.features, fcn32s.features):
        if (isinstance(l1, torch.nn.Conv2d) and
                isinstance(l2, torch.nn.Conv2d)):
            assert l1.weight.size() == l2.weight.size()
            assert l1.bias.size() == l2.bias.size()
            l2.weight.data = l1.weight.data
            l2.bias.data = l1.bias.data
    for i1, i2 in zip([1, 4], [0, 3]):
        l1 = vgg16.classifier[i1]
        l2 = fcn32s.classifier[i2]
        l2.weight.data = l1.weight.data.view(l2.weight.size())
        l2.bias.data = l1.bias.data.view(l2.bias.size())
    n_class = fcn32s.classifier[6].weight.size()[0]
    if copy_fc8:
        l1 = vgg16.classifier[6]
        l2 = fcn32s.classifier[6]
        l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
        l2.bias.data = l1.bias.data[:n_class]
    if init_upscore:
        # initialize upscore layer
        upscore = fcn32s.upscore[0]
        c1, c2, h, w = upscore.weight.data.size()
        assert c1 == c2 == n_class
        assert h == w
        weight = conv.get_upsample_filter(h)
        upscore.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
