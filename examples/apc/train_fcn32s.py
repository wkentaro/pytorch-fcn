#!/usr/bin/env python

import datetime
import os
import os.path as osp
import shlex
import shutil
import subprocess

import click
import pytz
import torch
import yaml

import torchfcn


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    hash = subprocess.check_output(shlex.split(cmd)).strip()
    return hash


here = osp.dirname(osp.abspath(__file__))


def load_config_file(config_file):
    # load config
    cfg = yaml.load(open(config_file))
    name = osp.splitext(osp.basename(config_file))[0]
    for k, v in cfg.items():
        name += '_%s-%s' % (k.upper(), str(v))
    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    name += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
    name += '_VCS-%s' % git_hash()
    # create out
    out = osp.join(here, 'logs', name)
    if not osp.exists(out):
        os.makedirs(out)
    shutil.copy(config_file, osp.join(out, 'config.yaml'))
    return cfg, out


@click.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--resume', type=click.Path(exists=True))
def main(config_file, resume):
    cfg, out = load_config_file(config_file)

    cuda = torch.cuda.is_available()

    batch_size = torch.cuda.device_count() * 3
    max_iter = cfg['max_iteration'] // batch_size

    torch.manual_seed(1)
    if cuda:
        torch.cuda.manual_seed(1)

    # 1. dataset

    cfg['dataset'] = cfg.get('dataset', 'v2')
    if cfg['dataset'] == 'v2':
        dataset_class = torchfcn.datasets.APC2016V2
    elif cfg['dataset'] == 'v3':
        dataset_class = torchfcn.datasets.APC2016V3
    else:
        raise ValueError('Unsupported dataset: %s' % cfg['dataset'])

    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        dataset_class(split='train', transform=True),
        batch_size=batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(
        dataset_class(split='valid', transform=True),
        batch_size=batch_size, shuffle=False, **kwargs)

    # 2. model

    n_class = len(train_loader.dataset.class_names)
    model = torchfcn.models.FCN32s(n_class=n_class, nodeconv=cfg['nodeconv'])
    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        vgg16 = torchfcn.models.VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16, copy_fc8=False, init_upscore=False)
    if cuda:
        if torch.cuda.device_count() == 1:
            model = model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # 3. optimizer

    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                             weight_decay=cfg['weight_decay'])
    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=valid_loader,
        out=out,
        max_iter=max_iter,
        interval_validate=cfg['interval_validate'],
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_epoch * len(train_loader)
    trainer.train()


if __name__ == '__main__':
    main()
