#!/usr/bin/env python

import multiprocessing
import os.path as osp

import torchfcn
import tqdm
import yaml


def do_work(args):
    index, data_id = args
    try:
        dataset._load_from_id(data_id)
    except Exception:
        return index


if __name__ == '__main__':
    dataset = torchfcn.datasets.apc.mit_training.APC2016mit_training

    pool = multiprocessing.Pool()
    tasks = list(enumerate(dataset._get_ids()))
    blacklist = []
    for index in tqdm.tqdm(pool.imap_unordered(do_work, tasks),
                           total=len(tasks), ncols=80):
        if index is not None:
            blacklist.append(index)

    data_dir = osp.join(osp.dirname(torchfcn.__file__), 'datasets/apc/data')
    with open(osp.join(data_dir, 'mit_training_blacklist.yaml'), 'w') as f:
        yaml.safe_dump(blacklist, f, default_flow_style=False)
