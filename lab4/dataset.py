from collections import defaultdict
from random import choice

import torchvision
from torch.utils.data import Dataset


class MNISTMetricDataset(Dataset):
    def __init__(self, root="/tmp/mnist/", split='train'):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))

        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]

    def _sample_negative(self, index):
        target_id = self.targets[index].item()
        classes_choice = list(self.target2indices.keys())
        classes_choice.remove(target_id)
        class_choice = choice(classes_choice)
        index_choice = choice(self.target2indices[class_choice])
        return index_choice

    def _sample_positive(self, index):
        target_id = self.targets[index].item()
        rand_index = choice(self.target2indices[target_id])
        return rand_index

    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)