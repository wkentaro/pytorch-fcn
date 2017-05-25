#!/usr/bin/env python

import os.path as osp

import torchfcn
import tqdm
import yaml


dataset = torchfcn.datasets.apc.mit_training.APC2016mit_training()

blacklist = []
for data_id in tqdm.tqdm(list(dataset._get_ids()), ncols=80):
    try:
        dataset._load_from_id(data_id)
    except Exception:
        blacklist.append(data_id)

data_dir = osp.join(osp.dirname(torchfcn.__file__), 'datasets/apc/data')

with open(osp.join(data_dir, 'mit_training_blacklist.yaml'), 'w') as f:
    yaml.safe_dump(blacklist, f, default_flow_style=False)
