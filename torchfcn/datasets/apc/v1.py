from base import APC2016Base
from jsk import APC2016jsk
from rbo import APC2016rbo


class APC2016V1(APC2016Base):

    def __init__(self, root, train=True, transform=False):
        self.datasets = [
            APC2016jsk(root, train, transform),
            APC2016rbo(root, train, transform),
        ]

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    @property
    def train(self):
        train = self.datasets[0].train
        assert all(d.train == train for d in self.datasets)
        return train

    @train.setter
    def train(self, value):
        for d in self.datasets:
            d.train = value

    def __getitem__(self, index):
        skipped = 0
        for dataset in self.datasets:
            current_index = index - skipped
            if current_index < len(dataset):
                return dataset[current_index]
            skipped += len(dataset)
