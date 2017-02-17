#!/usr/bin/env python

import argparse
import time

import chainer
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

t_start = time.time()
for i in xrange(n_eval):
    model(x)
elapsed_time = time.time() - t_start

print('Elapsed time: %.2f [s / %d evals]' % (elapsed_time, n_eval))
print('Hz: %.2f [hz]' % (n_eval / elapsed_time))


print('==> Testing FCN32s with PyTorch')
model = torchfcn.models.FCN32s()
model = model.cuda(device_id=gpu)

x_data = np.random.random((1, 3, 480, 640))
x = torch.autograd.Variable(torch.from_numpy(x_data).float(), volatile=True)
x = x.cuda(device_id=gpu)

t_start = time.time()
for i in xrange(n_eval):
    model(x)
elapsed_time = time.time() - t_start

print('Elapsed time: %.2f [s / %d evals]' % (elapsed_time, n_eval))
print('Hz: %.2f [hz]' % (n_eval / elapsed_time))
