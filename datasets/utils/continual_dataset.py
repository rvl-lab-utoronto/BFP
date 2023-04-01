# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from typing import Any, Tuple
from torchvision import datasets
import numpy as np
import torch.optim

class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    N_CLASSES = None
    TRANSFORM = None
    IMG_SIZE = None
    N_CHANNELS = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        # Training and testing loaders for tasks seen so far
        self.test_loaders = []
        self.train_loaders = []
        self.train_loader_all = None
        self.test_loader_all = None

        # Training and testing loaders up to a certain task
        self.test_loaders_up_to = []
        self.train_loaders_up_to = []

        self.args = args

        # For backward compactibility
        self.task_id = 0

    @abstractmethod
    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        Creates and returns training and testing dataset containing all data from all tasks.
        :return: a tuple containing the training and testing dataset
        """
        pass

    @property
    def train_loader(self) -> DataLoader:
        """
        Returns the training loader for the current task.
        :return: the training loader
        """
        print("ContinualDataset: Getting the train loader for task {}".format(self.task_id))
        return self.train_loaders[self.task_id]

    def build_data_loaders(self) -> None:
        self.train_loaders = []
        self.test_loaders = []
        for t in range(self.N_TASKS):
            # Here we need to Reinitialize the dataset as they will be modified in place
            train_dataset, test_dataset = self.get_datasets()
            c_start = t * self.N_CLASSES_PER_TASK
            c_end = (t + 1) * self.N_CLASSES_PER_TASK
            train_loader, test_loader = get_loader_given_classes(train_dataset, test_dataset,
                                                                 c_start, c_end, self.args)
            self.train_loaders.append(train_loader)
            self.test_loaders.append(test_loader)

        # Build training and testing dataloader for all tasks jointly
        self.train_loader_all = DataLoader(ConcatDataset([_.dataset for _ in self.train_loaders]),
            batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        self.test_loader_all = DataLoader(ConcatDataset([_.dataset for _ in self.test_loaders]),
            batch_size=self.args.batch_size, shuffle=False, num_workers=4)

    def build_data_loaders_up_to(self) -> None:
        self.train_loaders_up_to = []
        self.test_loaders_up_to = []
        for t in range(self.N_TASKS):
            train_dataset, test_dataset = self.get_datasets()
            c_start = 0
            c_end = (t + 1) * self.N_CLASSES_PER_TASK
            train_loader, test_loader = get_loader_given_classes(train_dataset, test_dataset,
                                                                    c_start, c_end, self.args)
            self.train_loaders_up_to.append(train_loader)
            self.test_loaders_up_to.append(test_loader)

    def mask_classes(self, outputs: torch.Tensor, t: int) -> None:
        """
        Given the output tensor, and the current task,
        masks the former by setting the responses for the other tasks at -inf.
        It is used to obtain the results for the task-il setting.
        :param outputs: the output tensor
        :param dataset: the continual dataset
        :param k: the task index
        """
        outputs[:, 0:t * self.N_CLASSES_PER_TASK] = -float('inf')
        outputs[:, (t + 1) * self.N_CLASSES_PER_TASK:
                self.N_TASKS * self.N_CLASSES_PER_TASK] = -float('inf')

    def get_data_loaders_by_task(self, t) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the given tasks.
        :param t: the task id (from 0 to N_TASKS - 1)
        :return: the training and test loaders
            The testing loader is a list of test loaders, one for each task seen so far.
        """
        return self.train_loaders[t], self.test_loaders[t]

    def get_data_loaders_up_to_task(self, t) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the given tasks.
        :param t: the task id (from 0 to N_TASKS - 1)
        :return: the training and test loaders
            The testing loader is a list of test loaders, one for each task seen so far.
        """
        return self.train_loaders_up_to[t], self.test_loaders_up_to[t]

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        """
        Returns the loss to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler:
        """
        Returns the scheduler to be used for to the current dataset.
        """
        pass

    @staticmethod
    def get_optimizer_scheduler(model, args: Namespace) -> Tuple[Any, Any]:
        """
        Returns the scheduler to be used for to the current dataset.
        """
        optimizer = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = None
        return optimizer, scheduler

    @staticmethod
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass

    @staticmethod
    def get_minibatch_size():
        pass


def get_loader_given_classes(train_dataset: datasets, test_dataset: datasets,
                             c_start:int, c_end:int, args: Namespace) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param cont_dataset: continual learning dataset
    :param c_start: start class (included)
    :param c_end: end class (not included)
    :return: train and test loaders
    """
    # Masks the data for the given classes
    train_mask = np.logical_and(np.array(train_dataset.targets) >= c_start,
        np.array(train_dataset.targets) < c_end)
    test_mask = np.logical_and(np.array(test_dataset.targets) >= c_start,
        np.array(test_dataset.targets) < c_end)

    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]

    ## Here it does not manipulate the labels
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader
