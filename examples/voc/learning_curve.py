#!/usr/bin/env python

from __future__ import division

import argparse
import os.path as osp

import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt
import pandas
import seaborn


def learning_curve(log_file):
    print('==> Plotting log file: %s' % log_file)

    df = pandas.read_csv(log_file)

    colors = ['red', 'green', 'blue', 'purple', 'orange']
    colors = seaborn.xkcd_palette(colors)

    plt.figure(figsize=(20, 6), dpi=500)

    row_min = df.min()
    row_max = df.max()

    # initialize DataFrame for train
    columns = [
        'epoch',
        'iteration',
        'train/loss',
        'train/acc',
        'train/acc_cls',
        'train/mean_iu',
        'train/fwavacc',
    ]
    df_train = df[columns]
    df_train = pandas.rolling_mean(df_train, window=10)
    df_train = df_train.dropna()
    iter_per_epoch = df_train.query('epoch == 1')['iteration'].values[0]
    df_train['epoch_detail'] = df_train['iteration'] / iter_per_epoch

    # initialize DataFrame for val
    columns = [
        'epoch',
        'iteration',
        'valid/loss',
        'valid/acc',
        'valid/acc_cls',
        'valid/mean_iu',
        'valid/fwavacc',
    ]
    df_valid = df[columns]
    df_valid = df_valid.dropna()
    iter_per_epoch = df_valid.query('epoch == 1')['iteration'].values[0]
    df_valid['epoch_detail'] = df_valid['iteration'] / iter_per_epoch

    data_frames = {'train': df_train, 'valid': df_valid}

    n_row = 2
    n_col = 3
    for i, split in enumerate(['train', 'valid']):
        df_split = data_frames[split]

        # loss
        plt.subplot(n_row, n_col, i * n_col + 1)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.plot(df_split['epoch_detail'], df_split['%s/loss' % split], '-',
                 markersize=1, color=colors[0], alpha=.5,
                 label='%s loss' % split)
        plt.xlim((0, row_max['epoch']))
        plt.ylim((min(row_min['train/loss'], row_min['valid/loss']),
                  max(row_max['train/loss'], row_max['valid/loss'])))
        plt.xlabel('epoch')
        plt.ylabel('%s loss' % split)

        # loss (log)
        plt.subplot(n_row, n_col, i * n_col + 2)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.semilogy(df_split['epoch_detail'], df_split['%s/loss' % split],
                     '-', markersize=1, color=colors[0], alpha=.5,
                     label='%s loss' % split)
        plt.xlim((0, row_max['epoch']))
        plt.ylim((min(row_min['train/loss'], row_min['valid/loss']),
                  max(row_max['train/loss'], row_max['valid/loss'])))
        plt.xlabel('epoch')
        plt.ylabel('%s loss (log)' % split)

        # lbl accuracy
        plt.subplot(n_row, n_col, i * n_col + 3)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.plot(df_split['epoch_detail'], df_split['%s/acc' % split],
                 '-', markersize=1, color=colors[1], alpha=.5,
                 label='%s accuracy' % split)
        plt.plot(df_split['epoch_detail'], df_split['%s/acc_cls' % split],
                 '-', markersize=1, color=colors[2], alpha=.5,
                 label='%s accuracy class' % split)
        plt.plot(df_split['epoch_detail'], df_split['%s/mean_iu' % split],
                 '-', markersize=1, color=colors[3], alpha=.5,
                 label='%s mean IU' % split)
        plt.plot(df_split['epoch_detail'], df_split['%s/fwavacc' % split],
                 '-', markersize=1, color=colors[4], alpha=.5,
                 label='%s fwav accuracy' % split)
        plt.legend()
        plt.xlim((0, row_max['epoch']))
        plt.ylim((0, 1))
        plt.xlabel('epoch')
        plt.ylabel('%s label accuracy' % split)

    out_file = osp.splitext(log_file)[0] + '.png'
    plt.savefig(out_file)
    print('==> Wrote figure to: %s' % out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file')
    args = parser.parse_args()

    log_file = args.log_file

    learning_curve(log_file)


if __name__ == '__main__':
    main()
