import os
import os.path as osp

import numpy as np
import skimage.io

from base import APC2016Base


class APC2016mit_training(APC2016Base):

    def __init__(self, transform=False):
        self._transform = transform
        self.dataset_dir = osp.expanduser('~/data/datasets/APC2016/training')
        self._ids = list(self._get_ids())

    def __len__(self):
        return len(self._ids)

    def _get_ids(self):
        for loc in ['shelf', 'tote']:
            loc_dir = osp.join(self.dataset_dir, loc)
            for cls_id, cls in enumerate(self.class_names):
                if cls_id == 0:  # background
                    continue
                cls_dir = osp.join(loc_dir, cls)
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

    def _load_from_id(self, data_id):
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
