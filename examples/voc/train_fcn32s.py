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
    parser.add_argument('--year', type=int, default=2012, choices=(2011, 2012),
                        help='Year of VOC dataset (default: 2012)')
    parser.add_argument('--resume')
    args = parser.parse_args()

    cuda = torch.cuda.is_available()

    year = args.year
    out = args.out
    resume = args.resume

    seed = 1
    max_iter = 100000

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # 1. dataset

    if year == 2011:
        dataset_class = torchfcn.datasets.VOC2011ClassSeg
    elif year == 2012:
        dataset_class = torchfcn.datasets.VOC2012ClassSeg
    else:
        raise ValueError

    root = osp.expanduser('~/data/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        dataset_class(root, train=True, transform=True),
        batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        dataset_class(root, train=False, transform=True),
        batch_size=1, shuffle=False, **kwargs)

    # 2. model

    model = torchfcn.models.FCN32s(n_class=21)
    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        pth_file = osp.expanduser('~/data/models/torch/vgg16-00b39a1b.pth')
        vgg16 = torchvision.models.vgg16()
        vgg16.load_state_dict(torch.load(pth_file))
        for l1, l2 in zip(vgg16.features, model.features):
            if (isinstance(l1, torch.nn.Conv2d) and
                    isinstance(l2, torch.nn.Conv2d)):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i1, i2 in zip([1, 4], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = model.segmenter[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        l1 = vgg16.classifier[6]
        l2 = model.segmenter[6]
        n_class = l2.weight.size()[0]
        l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
        l2.bias.data = l1.bias.data[:n_class]
        # initialize upscore layer
        upscore = model.segmenter[7]
        h, w, c1, c2 = upscore.weight.data.size()
        assert h == w
        weight = torchfcn.utils.get_upsample_filter(h)
        upscore.weight.data = weight.view(h, w, 1, 1).repeat(1, 1, c1, c2)
    if cuda:
        model = model.cuda()

    # 3. optimizer

    optimizer = optim.SGD(model.parameters(), lr=1e-10, momentum=0.99,
                          weight_decay=0.0005)

    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        out=out,
        max_iter=max_iter,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_epoch * len(train_loader)
    trainer.train()


if __name__ == '__main__':
    main()
