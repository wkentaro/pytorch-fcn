import glob
import os
import os.path as osp
import re

import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split

from base import APC2016Base


class APC2016rbo(APC2016Base):

    def __init__(self, root, train=True, transform=False):
        self.train = train
        self._transform = transform
        self.dataset_dir = osp.join(root, 'apc2016/APC2016rbo')
        data_ids = self._get_ids()
        ids_train, ids_val = train_test_split(
            data_ids, test_size=0.25, random_state=1234)
        self._ids = {'train': ids_train, 'val': ids_val}

    def __len__(self):
        split = 'train' if self.train else 'val'
        return len(self._ids[split])

    def _get_ids(self):
        ids = []
        for img_file in os.listdir(self.dataset_dir):
            if not re.match(r'^.*_[0-9]*_bin_[a-l].jpg$', img_file):
                continue
            data_id = osp.splitext(img_file)[0]
            ids.append(data_id)
        return ids

    def _load_from_id(self, data_id):
        img_file = osp.join(self.dataset_dir, data_id + '.jpg')
        img = scipy.misc.imread(img_file)
        # generate label from mask files
        lbl = np.zeros(img.shape[:2], dtype=np.int32)
        # shelf bin mask file
        shelf_bin_mask_file = osp.join(self.dataset_dir, data_id + '.pbm')
        shelf_bin_mask = scipy.misc.imread(shelf_bin_mask_file, mode='L')
        lbl[shelf_bin_mask < 127] = -1
        # object mask files
        mask_glob = osp.join(self.dataset_dir, data_id + '_*.pbm')
        for mask_file in glob.glob(mask_glob):
            mask_id = osp.splitext(osp.basename(mask_file))[0]
            mask = scipy.misc.imread(mask_file, mode='L')
            lbl_name = mask_id[len(data_id + '_'):]
            lbl_id = np.where(self.class_names == lbl_name)[0]
            lbl[mask > 127] = lbl_id
        return img, lbl

    def __getitem__(self, index):
        split = 'train' if self.train else 'val'
        data_id = self._ids[split][index]
        img, lbl = self._load_from_id(data_id)
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl
