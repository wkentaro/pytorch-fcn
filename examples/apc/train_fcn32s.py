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

    out = args.out
    resume = args.resume
    cuda = torch.cuda.is_available()

    seed = 1
    batch_size = torch.cuda.device_count() * 3
    max_iter = 100000 // batch_size

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
        if torch.cuda.device_count() == 1:
            model = model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # 3. optimizer

    optim = torch.optim.SGD(
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
