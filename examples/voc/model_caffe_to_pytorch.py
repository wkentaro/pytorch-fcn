#!/usr/bin/env python

import os.path as osp
import pkg_resources
import sys

import torch

# FIXME: must be after import torch
import caffe

import torchfcn


models = [
    ('fcn32s', 'FCN32s', []),
    ('fcn16s', 'FCN16s', []),
    ('fcn8s', 'FCN8s', []),
    ('fcn8s-atonce', 'FCN8sAtOnce', ['scale_pool4', 'scale_pool3']),
]


for name_lower, name_upper, blacklists in models:
    print('==> Loading caffe model of %s' % name_upper)
    pkg_root = pkg_resources.get_distribution('torchfcn').location
    sys.path.insert(
        0, osp.join(pkg_root, 'torchfcn/ext/fcn.berkeleyvision.org'))
    caffe_prototxt = osp.join(
        pkg_root,
        'torchfcn/ext/fcn.berkeleyvision.org/voc-%s/deploy.prototxt' %
        name_lower)
    caffe_model_path = osp.expanduser(
        '~/data/models/caffe/%s-heavy-pascal.caffemodel' % name_lower)
    caffe_model = caffe.Net(caffe_prototxt, caffe_model_path, caffe.TEST)

    torch_model = getattr(torchfcn.models, name_upper)()

    torch_model_params = torch_model.parameters()
    for name, p1 in caffe_model.params.iteritems():
        if name in blacklists:
            continue
        l2 = getattr(torch_model, name)
        p2 = l2.weight
        assert p1[0].data.shape == tuple(p2.data.size())
        print('%s: %s -> %s' % (name, p1[0].data.shape, p2.data.size()))
        p2.data = torch.from_numpy(p1[0].data)
        if len(p1) == 2:
            p2 = l2.bias
            assert p1[1].data.shape == tuple(p2.data.size())
            print('%s: %s -> %s' % (name, p1[1].data.shape, p2.data.size()))
            p2.data = torch.from_numpy(p1[1].data)

    torch_model_path = osp.expanduser(
        '~/data/models/pytorch/%s-heavy-pascal.pth' % name_lower)
    torch.save(torch_model.state_dict(), torch_model_path)
    print('==> Saved pytorch model: %s' % torch_model_path)
