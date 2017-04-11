#!/usr/bin/env python

import argparse
import time

import numpy as np


def bench_chainer(gpu, times):
    import chainer
    import fcn
    print('==> Testing FCN32s with Chainer')
    chainer.cuda.get_device(gpu).use()

    x_data = np.random.random((1, 3, 480, 640)).astype(np.float32)
    x_data = chainer.cuda.to_gpu(x_data)
    x = chainer.Variable(x_data, volatile=True)

    model = fcn.models.FCN32s()
    model.train = False
    model.to_gpu()

    for i in xrange(5):
        model(x)
    chainer.cuda.Stream().synchronize()
    t_start = time.time()
    for i in xrange(times):
        model(x)
    chainer.cuda.Stream().synchronize()
    elapsed_time = time.time() - t_start

    print('Elapsed time: %.2f [s / %d evals]' % (elapsed_time, times))
    print('Hz: %.2f [hz]' % (times / elapsed_time))


def bench_pytorch(gpu, times):
    import torch
    import torchfcn.models
    print('==> Testing FCN32s with PyTorch')
    torch.cuda.set_device(gpu)

    model = torchfcn.models.FCN32s()
    model.eval()
    model = model.cuda()

    x_data = np.random.random((1, 3, 480, 640))
    x = torch.autograd.Variable(torch.from_numpy(x_data).float(),
                                volatile=True)
    x = x.cuda()

    for i in xrange(5):
        model(x)
    torch.cuda.synchronize()
    t_start = time.time()
    for i in xrange(times):
        model(x)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    print('Elapsed time: %.2f [s / %d evals]' % (elapsed_time, times))
    print('Hz: %.2f [hz]' % (times / elapsed_time))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--times', type=int, default=1000)
    args = parser.parse_args()

    print('==> Running on GPU: %d to evaluate %d times' %
          (args.gpu, args.times))
    bench_chainer(args.gpu, args.times)
    bench_pytorch(args.gpu, args.times)


if __name__ == '__main__':
    main()
