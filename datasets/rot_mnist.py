# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torchvision.transforms as transforms
from datasets.transforms.rotation import Rotation
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from backbone.MNISTMLP import MNISTMLP
import torch.nn.functional as F
from datasets.perm_mnist import store_mnist_loaders
from datasets.utils.continual_dataset import ContinualDataset


class RotatedMNIST(ContinualDataset):
    NAME = 'rot-mnist'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 20
    N_CLASSES = 10
    TRANSFORM = None
    IMG_SIZE = (28,28)
    N_CHANNELS = 1

    def get_data_loaders(self):
        transform = transforms.Compose((Rotation(), transforms.ToTensor()))
        train, test = store_mnist_loaders(transform, self)
        return train, test

    def build_data_loaders(self) -> None:
        self.train_loaders = []
        self.test_loaders = []

        for i in range(self.N_TASKS):
            train, test = self.get_data_loaders()
            self.train_loaders.append(train)
            self.test_loaders.append(test)

        # Build training and testing dataloader for all tasks jointly
        self.train_loader_all = DataLoader(ConcatDataset([_.dataset for _ in self.train_loaders]),
            batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        self.test_loader_all = DataLoader(ConcatDataset([_.dataset for _ in self.test_loaders]),
            batch_size=self.args.batch_size, shuffle=False, num_workers=4)

    def build_data_loaders_up_to(self) -> None:
        assert len(self.train_loaders) > 0
        assert len(self.test_loaders) > 0
        
        self.train_loaders_up_to = []
        self.test_loaders_up_to = []

        for t in range(self.N_TASKS):
            self.train_loaders_up_to.append(DataLoader(ConcatDataset([_.dataset for _ in self.train_loaders[:t+1]]),
                batch_size=self.args.batch_size, shuffle=True, num_workers=4))
            self.test_loaders_up_to.append(DataLoader(ConcatDataset([_.dataset for _ in self.test_loaders[:t+1]]),
                batch_size=self.args.batch_size, shuffle=False, num_workers=4))

    @staticmethod
    def get_backbone():
        return MNISTMLP(28 * 28, RotatedMNIST.N_CLASSES_PER_TASK)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args):
        return None