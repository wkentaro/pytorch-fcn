#!/usr/bin/env python

import os
import os.path as osp
import shutil

import click
import torch
import yaml

import torchfcn


def load_config_file(config_file):
    # load config
    cfg = yaml.load(open(config_file))
    name = osp.splitext(osp.basename(config_file))[0]
    for k, v in cfg.items():
        name += '_%s-%s' % (k.upper(), str(v))
    # create out
    out = osp.join(here, 'logs', name)
    if not osp.exists(out):
        os.makedirs(out)
    shutil.copy(config_file, osp.join(out, 'config.yaml'))
    return cfg, out


here = osp.dirname(osp.abspath(__file__))


@click.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--resume', type=click.Path(exists=True))
def main(config_file, resume):
    cfg, out = load_config_file(config_file)

    cuda = torch.cuda.is_available()

    torch.manual_seed(1)
    if cuda:
        torch.cuda.manual_seed(1)

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

    model = torchfcn.models.FCN32s(n_class=21, nodeconv=cfg['nodeconv'])
    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        vgg16 = torchfcn.models.VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16, init_upscore=False)
    if cuda:
        model = model.cuda()

    # 3. optimizer

    optim = torch.optim.SGD(
        model.parameters(),
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
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_epoch * len(train_loader)
    trainer.train()


if __name__ == '__main__':
    main()
