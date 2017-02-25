#!/usr/bin/env python

import os.path as osp
import pkg_resources
import sys

import caffe
import torch
import torchfcn


print('==> Loading FCN32s model from Caffe')
pkg_root = pkg_resources.get_distribution('torchfcn').location
sys.path.insert(0, osp.join(pkg_root, 'torchfcn/ext/fcn.berkeleyvision.org'))
caffe_prototxt = osp.join(
    pkg_root, 'torchfcn/ext/fcn.berkeleyvision.org/voc-fcn32s/deploy.prototxt')
caffe_model_path = osp.expanduser(
    '~/data/models/caffe/fcn32s-heavy-pascal.caffemodel')
caffe_model = caffe.Net(caffe_prototxt, caffe_model_path, caffe.TEST)

torch_model = torchfcn.models.FCN32s()

torch_model_params = torch_model.parameters()
for name, p1 in caffe_model.params.iteritems():
    p2 = torch_model_params.next()
    print('%s: %s -> %s' % (name, p1[0].data.shape, p2.data.size()))
    p2.data = torch.from_numpy(p1[0].data)
    if len(p1) == 2:
        p2 = torch_model_params.next()
        print('%s: %s -> %s' % (name, p1[1].data.shape, p2.data.size()))
        p2.data = torch.from_numpy(p1[1].data)

torch_model_path = osp.expanduser(
    '~/data/models/torch/fcn32s-heavy-pascal.pth')
print('==> Saving FCN32s PyTorch model to: %s' % torch_model_path)
torch.save(torch_model.state_dict(), torch_model_path)
