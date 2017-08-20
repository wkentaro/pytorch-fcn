#!/usr/bin/env python

import argparse
import time

import numpy as np
import six


def bench_chainer(gpu, times, dynamic_input=False):
    import chainer
    import fcn
    print('==> Testing FCN32s with Chainer')
    chainer.cuda.get_device(gpu).use()

    chainer.config.train = False
    chainer.config.enable_backprop = False

    if dynamic_input:
        x_data = np.random.random((1, 3, 480, 640)).astype(np.float32)
        x_data = chainer.cuda.to_gpu(x_data)
        x1 = chainer.Variable(x_data)
        x_data = np.random.random((1, 3, 640, 480)).astype(np.float32)
        x_data = chainer.cuda.to_gpu(x_data)
        x2 = chainer.Variable(x_data)
    else:
        x_data = np.random.random((1, 3, 480, 640)).astype(np.float32)
        x_data = chainer.cuda.to_gpu(x_data)
        x1 = chainer.Variable(x_data)

    model = fcn.models.FCN32s()
    model.train = False
    model.to_gpu()

    for i in six.moves.range(5):
        model(x1)
    chainer.cuda.Stream().synchronize()
    t_start = time.time()
    for i in six.moves.range(times):
        if dynamic_input:
            if i % 2 == 1:
                model(x1)
            else:
                model(x2)
        else:
            model(x1)
    chainer.cuda.Stream().synchronize()
    elapsed_time = time.time() - t_start

    print('Elapsed time: %.2f [s / %d evals]' % (elapsed_time, times))
    print('Hz: %.2f [hz]' % (times / elapsed_time))


def bench_pytorch(gpu, times, dynamic_input=False):
    import torch
    import torchfcn.models
    print('==> Testing FCN32s with PyTorch')
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = not dynamic_input

    model = torchfcn.models.FCN32s()
    model.eval()
    model = model.cuda()

    if dynamic_input:
        x_data = np.random.random((1, 3, 480, 640))
        x1 = torch.autograd.Variable(torch.from_numpy(x_data).float(),
                                     volatile=True).cuda()
        x_data = np.random.random((1, 3, 640, 480))
        x2 = torch.autograd.Variable(torch.from_numpy(x_data).float(),
                                     volatile=True).cuda()
    else:
        x_data = np.random.random((1, 3, 480, 640))
        x1 = torch.autograd.Variable(torch.from_numpy(x_data).float(),
                                     volatile=True).cuda()

    for i in six.moves.range(5):
        model(x1)
    torch.cuda.synchronize()
    t_start = time.time()
    for i in six.moves.range(times):
        if dynamic_input:
            if i % 2 == 1:
                model(x1)
            else:
                model(x2)
        else:
            model(x1)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    print('Elapsed time: %.2f [s / %d evals]' % (elapsed_time, times))
    print('Hz: %.2f [hz]' % (times / elapsed_time))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--times', type=int, default=1000)
    parser.add_argument('--dynamic-input', action='store_true')
    args = parser.parse_args()

    print('==> Benchmark: gpu=%d, times=%d, dynamic_input=%s' %
          (args.gpu, args.times, args.dynamic_input))
    bench_chainer(args.gpu, args.times, args.dynamic_input)
    bench_pytorch(args.gpu, args.times, args.dynamic_input)


if __name__ == '__main__':
    main()
