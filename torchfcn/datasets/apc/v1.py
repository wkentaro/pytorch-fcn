from base import APC2016Base
from jsk import APC2016jsk
from rbo import APC2016rbo


class APC2016V1(APC2016Base):

    def __init__(self, split='train', transform=False):
        self.datasets = [
            APC2016jsk(split, transform),
            APC2016rbo(split, transform),
        ]

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    @property
    def split(self):
        split = self.datasets[0].split
        assert all(d.split == split for d in self.datasets)
        return split

    @split.setter
    def split(self, value):
        for d in self.datasets:
            d.split = value

    def __getitem__(self, index):
        skipped = 0
        for dataset in self.datasets:
            current_index = index - skipped
            if current_index < len(dataset):
                return dataset[current_index]
            skipped += len(dataset)
