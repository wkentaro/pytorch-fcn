import matplotlib.pyplot as plt
from nose.tools import assert_true
import numpy as np
import skimage.data
import torch

from torchfcn.models.fcn32s import get_upsampling_weight


def test_get_upsampling_weight():
    src = skimage.data.coffee()
    x = src.transpose(2, 0, 1)
    x = x[np.newaxis, :, :, :]
    x = torch.from_numpy(x).float()
    x = torch.autograd.Variable(x)

    in_channels = 3
    out_channels = 3
    kernel_size = 4

    m = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride=2, bias=False)
    m.weight.data = get_upsampling_weight(
        in_channels, out_channels, kernel_size)

    y = m(x)

    y = y.data.numpy()
    y = y[0]
    y = y.transpose(1, 2, 0)
    dst = y.astype(np.uint8)

    assert_true(abs(src.shape[0] * 2 - dst.shape[0]) <= 2)
    assert_true(abs(src.shape[1] * 2 - dst.shape[1]) <= 2)

    return src, dst


if __name__ == '__main__':
    src, dst = test_get_upsampling_weight()
    plt.subplot(121)
    plt.imshow(src)
    plt.title('x1: {}'.format(src.shape))
    plt.subplot(122)
    plt.imshow(dst)
    plt.title('x2: {}'.format(dst.shape))
    plt.show()
