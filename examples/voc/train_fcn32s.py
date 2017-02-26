#!/usr/bin/env python

import argparse
import os.path as osp

import torch
import torchvision

import torchfcn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out')
    parser.add_argument('--resume')
    args = parser.parse_args()

    cuda = torch.cuda.is_available()

    out = args.out
    resume = args.resume

    seed = 1
    max_iter = 100000

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # 1. dataset

    root = osp.expanduser('~/data/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.SBDClassSeg(root, split='train', transform=True),
        batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.VOC2011ClassSeg(
            root, split='seg11valid', transform=True),
        batch_size=1, shuffle=False, **kwargs)

    # 2. model

    model = torchfcn.models.FCN32s(n_class=21)
    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        pth_file = osp.expanduser('~/data/models/torch/vgg16-00b39a1b.pth')
        vgg16 = torchvision.models.vgg16()
        vgg16.load_state_dict(torch.load(pth_file))
        torchfcn.utils.copy_params_vgg16_to_fcn32s(vgg16, model)
    if cuda:
        model = model.cuda()

    # 3. optimizer

    # FIXME: Per-parameter options not work? No loss discreasing.
    # conv_weights = []
    # conv_biases = []
    # for l in model.features:
    #     for i, param in enumerate(l.parameters()):
    #         if i == 0:
    #             conv_weights.append(param)
    #         elif i == 1:
    #             conv_biases.append(param)
    #         else:
    #             raise ValueError
    optim = torch.optim.SGD(
        # FIXME: Per-parameter options not work? No loss discreasing.
        # [
        #     {'params': conv_weights},
        #     {'params': conv_biases, 'lr': 2e-10, 'weight_decay': 0},
        #     {'params': model.upscore.parameters(), 'lr': 0},  # deconv
        # ],
        model.parameters(),
        lr=1e-10, momentum=0.99, weight_decay=0.0005)
    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
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
