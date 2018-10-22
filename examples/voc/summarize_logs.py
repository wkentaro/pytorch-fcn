#!/usr/bin/env python

import os
import os.path as osp

import pandas as pd
import tabulate
import yaml


def main():
    logs_dir = 'logs'

    headers = [
        'name',
        'model',
        'git_hash',
        'pretrained_model',
        'epoch',
        'iteration',
        'valid/mean_iu',
    ]
    rows = []
    for log in os.listdir(logs_dir):
        log_dir = osp.join(logs_dir, log)
        if not osp.isdir(log_dir):
            continue
        try:
            log_file = osp.join(log_dir, 'log.csv')
            df = pd.read_csv(log_file)
            columns = [c for c in df.columns if not c.startswith('train')]
            df = df[columns]
            df = df.set_index(['epoch', 'iteration'])
            index_best = df['valid/mean_iu'].idxmax()
            row_best = df.loc[index_best].dropna()

            with open(osp.join(log_dir, 'config.yaml')) as f:
                config = yaml.load(f)
        except Exception:
            continue
        rows.append([
            osp.join(logs_dir, log),
            config['model'],
            config['git_hash'],
            config.get('pretrained_model', None),
            row_best.index[0][0],
            row_best.index[0][1],
            100 * row_best['valid/mean_iu'].values[0],
        ])
    rows.sort(key=lambda x: x[-1], reverse=True)
    print(tabulate.tabulate(rows, headers=headers))


if __name__ == '__main__':
    main()
