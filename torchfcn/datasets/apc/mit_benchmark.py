import itertools
import os
import os.path as osp

import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split

from base import APC2016Base


def ids_from_scene_dir(scene_dir, empty_scene_dir):
    for i_frame in itertools.count():
        empty_file = osp.join(
            empty_scene_dir, 'frame-{:06}.color.png'.format(i_frame))
        rgb_file = osp.join(
            scene_dir, 'frame-{:06}.color.png'.format(i_frame))
        segm_file = osp.join(
            scene_dir, 'segm/frame-{:06}.segm.png'.format(i_frame))
        if not (osp.exists(rgb_file) and osp.exists(segm_file)):
            break
        data_id = (empty_file, rgb_file, segm_file)
        yield data_id


def bin_id_from_scene_dir(scene_dir):
    caminfo = open(osp.join(scene_dir, 'cam.info.txt')).read()
    loc = caminfo.splitlines()[0].split(': ')[-1]
    if loc == 'shelf':
        bin_id = caminfo.splitlines()[1][-1]
    else:
        bin_id = 'tote'
    return bin_id


class APC2016mit_benchmark(APC2016Base):

    def __init__(self, root, train=True, transform=False):
        self.train = train
        self._transform = transform
        self.dataset_dir = osp.join(root, 'apc2016/benchmark')
        data_ids = self._get_ids()
        ids_train, ids_val = train_test_split(
            data_ids, test_size=0.25, random_state=1234)
        self._ids = {'train': ids_train, 'val': ids_val}

    def __len__(self):
        split = 'train' if self.train else 'val'
        return len(self._ids[split])

    def _get_ids_from_loc_dir(self, env, loc_dir):
        assert env in ('office', 'warehouse')
        loc = osp.basename(loc_dir)
        data_ids = []
        for scene_dir in os.listdir(loc_dir):
            scene_dir = osp.join(loc_dir, scene_dir)
            bin_id = bin_id_from_scene_dir(scene_dir)
            empty_dir = osp.join(
                self.dataset_dir, env, 'empty', loc, 'scene-{}'.format(bin_id))
            data_ids += list(ids_from_scene_dir(scene_dir, empty_dir))
        return data_ids

    def _get_ids(self):
        data_ids = []
        # office
        contain_dir = osp.join(self.dataset_dir, 'office/test')
        for loc in ['shelf', 'tote']:
            loc_dir = osp.join(contain_dir, loc)
            data_ids += self._get_ids_from_loc_dir('office', loc_dir)
        # warehouse
        contain_dir = osp.join(self.dataset_dir, 'warehouse')
        for sub in ['practice', 'competition']:
            sub_contain_dir = osp.join(contain_dir, sub)
            for loc in ['shelf', 'tote']:
                loc_dir = osp.join(sub_contain_dir, loc)
                data_ids += self._get_ids_from_loc_dir('warehouse', loc_dir)
        return data_ids

    def _load_from_id(self, data_id):
        empty_file, rgb_file, segm_file = data_id
        img = scipy.misc.imread(rgb_file, mode='RGB')
        img_empty = scipy.misc.imread(empty_file, mode='RGB')
        # Label value is multiplied by 9:
        #   ex) 0: 0/6=0 (background), 54: 54/6=9 (dasani_bottle_water)
        lbl = scipy.misc.imread(segm_file, mode='L') / 6
        lbl = lbl.astype(np.int32)
        img_empty[lbl > 0] = img[lbl > 0]
        return img_empty, lbl

    def __getitem__(self, index):
        split = 'train' if self.train else 'val'
        data_id = self._ids[split][index]
        img, lbl = self._load_from_id(data_id)
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl
