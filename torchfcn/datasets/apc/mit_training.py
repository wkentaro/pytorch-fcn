import os
import os.path as osp

import numpy as np
import skimage.io
import yaml

from base import APC2016Base


here = osp.dirname(osp.abspath(__file__))


class APC2016mit_training(APC2016Base):

    dataset_dir = osp.expanduser('~/data/datasets/APC2016/training')

    def __init__(self, transform=False):
        self._transform = transform
        # drop by blacklist
        self._ids = []
        with open(osp.join(here, 'data/mit_training_blacklist.yaml')) as f:
            blacklist = yaml.load(f)
        for index, data_id in enumerate(self._get_ids()):
            if index in blacklist:
                print('WARNING: skipping index=%d data' % index)
                continue
            self._ids.append(data_id)

    def __len__(self):
        return len(self._ids)

    @classmethod
    def _get_ids(cls):
        for loc in ['shelf', 'tote']:
            loc_dir = osp.join(cls.dataset_dir, loc)
            for cls_id, cls_name in enumerate(cls.class_names):
                if cls_id == 0:  # background
                    continue
                cls_dir = osp.join(loc_dir, cls_name)
                scene_dir_empty = osp.join(cls_dir, 'scene-empty')
                for scene_dir in os.listdir(cls_dir):
                    scene_dir = osp.join(cls_dir, scene_dir)
                    for frame_id in xrange(0, 18):
                        empty_file = osp.join(
                            scene_dir_empty, 'frame-%06d.color.png' % frame_id)
                        rgb_file = osp.join(
                            scene_dir, 'frame-%06d.color.png' % frame_id)
                        mask_file = osp.join(
                            scene_dir, 'masks',
                            'frame-%06d.mask.png' % frame_id)
                        if osp.exists(rgb_file) and osp.exists(mask_file):
                            yield empty_file, rgb_file, mask_file, cls_id

    @staticmethod
    def _load_from_id(data_id):
        empty_file, rgb_file, mask_file, cls_id = data_id
        img = skimage.io.imread(rgb_file)
        img_empty = skimage.io.imread(empty_file)
        mask = skimage.io.imread(mask_file, as_grey=True) >= 0.5
        lbl = np.zeros(mask.shape, dtype=np.int32)
        lbl[mask] = cls_id
        img_empty[mask] = img[mask]
        return img_empty, lbl

    def __getitem__(self, index):
        data_id = self._ids[index]
        img, lbl = self._load_from_id(data_id)
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl
