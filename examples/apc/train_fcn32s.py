#!/usr/bin/env python

import argparse
import os.path as osp

# FIXME: torch -> scipy.misc raises SEGV
import scipy.misc  # NOQA

import torch
import torch.optim as optim
import torchvision

import torchfcn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out')
    args = parser.parse_args()

    out = args.out
    cuda = torch.cuda.is_available()

    seed = 1
    batch_size = 1
    max_iter = 100000

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # 1. dataset

    root = osp.expanduser('~/data/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.APC2016V2(root, train=True, transform=True),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.APC2016V2(root, train=False, transform=True),
        batch_size=batch_size, shuffle=False, **kwargs)

    # 2. model

    n_class = len(train_loader.dataset.class_names)
    model = torchfcn.models.FCN32s(n_class=n_class)
    pth_file = osp.expanduser('~/data/models/torch/vgg16-00b39a1b.pth')
    vgg16 = torchvision.models.vgg16()
    vgg16.load_state_dict(torch.load(pth_file))
    for l1, l2 in zip(vgg16.features, model.features):
        if isinstance(l1, torch.nn.Conv2d) and isinstance(l2, torch.nn.Conv2d):
            assert l1.weight.size() == l2.weight.size()
            assert l1.bias.size() == l2.bias.size()
            l2.weight.data = l1.weight.data
            l2.bias.data = l1.bias.data
    for i1, i2 in zip([1, 4], [0, 3]):
        l1 = vgg16.classifier[i1]
        l2 = model.classifier[i2]
        l2.weight.data = l1.weight.data.view(l2.weight.size())
        l2.bias.data = l1.bias.data.view(l2.bias.size())
    if cuda:
        if torch.cuda.device_count() == 1:
            model = model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # 3. optimizer

    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0005)

    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        out=out,
        max_iter=max_iter,
    )
    trainer.train()


if __name__ == '__main__':
    main()
