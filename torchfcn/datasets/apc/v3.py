from base import APC2016Base
from jsk import APC2016jsk
from mit_benchmark import APC2016mit_benchmark
from mit_training import APC2016mit_training
from rbo import APC2016rbo


class APC2016V3(APC2016Base):

    def __init__(self, split, transform):
        if split == 'train':
            self.datasets = [
                APC2016mit_training(transform),
                APC2016jsk('all', transform),
                APC2016rbo('all', transform),
            ]
        elif split == 'valid':
            self.datasets = [
                APC2016mit_benchmark('all', transform),
            ]
        else:
            raise ValueError('Unsupported split: %s' % split)

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    @property
    def split(self):
        raise RuntimeError('Not supported.')

    @split.setter
    def split(self, value):
        raise RuntimeError('Not supported.')

    def __getitem__(self, index):
        skipped = 0
        for dataset in self.datasets:
            current_index = index - skipped
            if current_index < len(dataset):
                return dataset[current_index]
            skipped += len(dataset)
