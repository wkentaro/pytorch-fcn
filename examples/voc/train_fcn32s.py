#!/usr/bin/env python

import datetime
import os
import os.path as osp
import shlex
import shutil
import subprocess

import click
import numpy as np
import pytz
import torch
import yaml

import torchfcn


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    hash = subprocess.check_output(shlex.split(cmd)).strip()
    return hash


def load_config_file(config_file):
    # load config
    cfg = yaml.load(open(config_file))
    name = osp.splitext(osp.basename(config_file))[0]
    for k, v in cfg.items():
        name += '_%s-%s' % (k.upper(), str(v))
    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    name += '_VCS-%s' % git_hash()
    name += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
    # create out
    out = osp.join(here, 'logs', name)
    if not osp.exists(out):
        os.makedirs(out)
    shutil.copy(config_file, osp.join(out, 'config.yaml'))
    return cfg, out


def get_parameters(model, bias=False):
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        # elif isinstance(m, nn.ConvTranspose2d):
        #     if not bias:
        #         yield m.weight


here = osp.dirname(osp.abspath(__file__))


@click.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--resume', type=click.Path(exists=True))
def main(config_file, resume):
    cfg, out = load_config_file(config_file)

    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

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
    start_iteration = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        vgg16_fcn32s = torchfcn.models.FCN32s(n_class=21)
        vgg16_fcn32s.load_state_dict(torch.load(osp.expanduser('~/data/models/torch/vgg16-fcn32s.pth')))
        model.copy_params_from_vgg16(vgg16_fcn32s, copy_fc8=False)
    if cuda:
        model = model.cuda()

    # 3. optimizer

    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': cfg['lr'] * 2, 'weight_decay': 0},
        ],
        lr=cfg['lr'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'])
    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=out,
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(train_loader)),
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
