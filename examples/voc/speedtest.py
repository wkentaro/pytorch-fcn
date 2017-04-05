#!/usr/bin/env python

import argparse
import time

import chainer
import cupy
import fcn
import numpy as np

import torch
import torchfcn.models


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--times', type=int, default=1000)
args = parser.parse_args()

gpu = args.gpu
n_eval = args.times

print('==> Running on GPU: %d to evaluate %d times' % (gpu, n_eval))


print('==> Testing FCN32s with Chainer')

x_data = np.random.random((1, 3, 480, 640)).astype(np.float32)
x_data = chainer.cuda.to_gpu(x_data, device=gpu)
x = chainer.Variable(x_data, volatile=True)

model = fcn.models.FCN32s()
model.train = False
model.to_gpu(device=gpu)
chainer.cuda.set_max_workspace_size(2 ** 30)

for i in range(5):
    model(x)
chainer.cuda.to_cpu(model.score.data)
start = cupy.cuda.Event()
end = cupy.cuda.Event()
start.record()
for i in xrange(n_eval):
    model(x)
end.record()
end.synchronize()
elapsed_time = 1. / 1000 * cupy.cuda.get_elapsed_time(start, end)  # seconds

print('Elapsed time: %.2f [s / %d evals]' % (elapsed_time, n_eval))
print('Hz: %.2f [hz]' % (n_eval / elapsed_time))


print('==> Testing FCN32s with PyTorch')
model = torchfcn.models.FCN32s()
model = model.cuda(device_id=gpu)

x_data = np.random.random((1, 3, 480, 640))
x = torch.autograd.Variable(torch.from_numpy(x_data).float(), volatile=True)
x = x.cuda(device_id=gpu)

for i in xrange(5):
    y = model(x)
start = cupy.cuda.Event()
end = cupy.cuda.Event()
start.record()
for i in xrange(n_eval):
    y = model(x)
end.record()
end.synchronize()
elapsed_time = 1. / 1000 * cupy.cuda.get_elapsed_time(start, end)  # seconds

print('Elapsed time: %.2f [s / %d evals]' % (elapsed_time, n_eval))
print('Hz: %.2f [hz]' % (n_eval / elapsed_time))
