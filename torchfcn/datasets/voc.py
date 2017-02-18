#!/usr/bin/env python

import collections
import os.path as osp

import numpy as np
import PIL.Image
import torch
from torch.utils import data


class VOC2012ClassSeg(data.Dataset):

    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, root, train=True, transform=False):
        self.root = root
        self.train = train
        self._transform = transform

        dataset_dir = osp.join(
            self.root, 'VOCdevkit/VOC2012')
        self.files = collections.defaultdict(list)
        for data_type in ['train', 'val']:
            imgsets_file = osp.join(
                dataset_dir, 'ImageSets/Segmentation/%s.txt' % data_type)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
                lbl_file = osp.join(
                    dataset_dir, 'SegmentationClass/%s.png' % did)
                self.files[data_type].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        data_type = 'train' if self.train else 'val'
        return len(self.files[data_type])

    def __getitem__(self, index):
        data_type = 'train' if self.train else 'val'
        data_file = self.files[data_type][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl
