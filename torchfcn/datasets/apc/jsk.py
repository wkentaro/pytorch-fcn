import os
import os.path as osp

import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split

from base import APC2016Base


class APC2016jsk(APC2016Base):

    def __init__(self, root, train=True, transform=False):
        self.train = train
        self._transform = transform
        self.dataset_dir = osp.join(root, 'apc2016/annotated')
        data_ids = self._get_ids()
        ids_train, ids_val = train_test_split(
            data_ids, test_size=0.25, random_state=1234)
        self._ids = {'train': ids_train, 'val': ids_val}

    def __len__(self):
        split = 'train' if self.train else 'val'
        return len(self._ids[split])

    def _get_ids(self):
        ids = []
        for data_id in os.listdir(self.dataset_dir):
            ids.append(data_id)
        return ids

    def _load_from_id(self, data_id):
        img_file = osp.join(self.dataset_dir, data_id, 'image.png')
        img = scipy.misc.imread(img_file)
        lbl_file = osp.join(self.dataset_dir, data_id, 'label.png')
        lbl = scipy.misc.imread(lbl_file, mode='L')
        lbl = lbl.astype(np.int32)
        lbl[lbl == 255] = -1
        return img, lbl

    def __getitem__(self, index):
        split = 'train' if self.train else 'val'
        data_id = self._ids[split][index]
        img, lbl = self._load_from_id(data_id)
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl
