#!/usr/bin/env python

import argparse

# FIXME: torch -> scipy.misc raises SEGV
import scipy.misc  # NOQA

import torch
import torch.optim as optim
import torchvision

import torchfcn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--year', type=int, default=2012, choices=(2011, 2012),
                        help='Year of VOC dataset (default: 2012)')
    args = parser.parse_args()

    gpu = args.gpu
    year = args.year
    out = args.out

    seed = 1
    max_iter = 100000

    torch.manual_seed(seed)
    if gpu >= 0:
        torch.cuda.manual_seed(seed)

    torch.cuda.set_device(gpu)

    # 1. dataset

    if year == 2011:
        dataset_class = torchfcn.datasets.VOC2011ClassSeg
    elif year == 2012:
        dataset_class = torchfcn.datasets.VOC2012ClassSeg
    else:
        raise ValueError

    root = osp.expanduser('~/data/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if gpu >= 0 else {}
    train_loader = torch.utils.data.DataLoader(
        dataset_class(root, train=True, transform=True),
        batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        dataset_class(root, train=False, transform=True),
        batch_size=1, shuffle=False, **kwargs)

    # 2. model

    model = torchfcn.models.FCN32s(n_class=21)
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
        l2 = model.segmenter[i2]
        l2.weight.data = l1.weight.data.view(l2.weight.size())
        l2.bias.data = l1.bias.data.view(l2.bias.size())
    if gpu >= 0:
        model = model.cuda()

    # 3. optimizer

    optimizer = optim.SGD(model.parameters(), lr=1e-10, momentum=0.99,
                          weight_decay=0.0005)

    trainer = torchfcn.Trainer(
        device_ids=[gpu],
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
