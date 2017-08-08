#!/usr/bin/env python

import os.path as osp
import pkg_resources
import sys

import torch
import caffe  # NOQA: must be after import caffe
import torchfcn


for size_reactive in [32, 16, 8]:
    print('==> Loading caffe model of FCN%ds' % size_reactive)
    pkg_root = pkg_resources.get_distribution('torchfcn').location
    sys.path.insert(
        0, osp.join(pkg_root, 'torchfcn/ext/fcn.berkeleyvision.org'))
    caffe_prototxt = osp.join(
        pkg_root,
        'torchfcn/ext/fcn.berkeleyvision.org/voc-fcn%ds/deploy.prototxt' %
        size_reactive)
    caffe_model_path = osp.expanduser(
        '~/data/models/caffe/fcn%ss-heavy-pascal.caffemodel' % size_reactive)
    caffe_model = caffe.Net(caffe_prototxt, caffe_model_path, caffe.TEST)

    torch_model = getattr(torchfcn.models, 'FCN%ds' % size_reactive)()

    torch_model_params = torch_model.parameters()
    for name, p1 in caffe_model.params.iteritems():
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
        '~/data/models/pytorch/fcn%ss-heavy-pascal.pth' % size_reactive)
    torch.save(torch_model.state_dict(), torch_model_path)
    print('==> Saved pytorch model: %s' % torch_model_path)
