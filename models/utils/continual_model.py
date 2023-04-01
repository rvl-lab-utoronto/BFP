# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.optim import SGD
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.task_id = 0

        assert not (self.args.freeze_feature and self.args.freeze_classifier) 
        if self.args.freeze_feature:
            self.net.freeze_feature()
        if self.args.freeze_classifier:
            self.net.freeze_classifier()

        self.device = get_device()

    def reset_optimizers(self, dataset):
        self.opt, self.scheduler = dataset.get_optimizer_scheduler(self, self.args)
    
    def begin_task(self, dataset, t=0, start_epoch=0):
        self.task_id = t
        
        if start_epoch == 0:
            self.reset_optimizers(dataset)

        # For backward compactibility
        dataset.task_id = t

    def end_task(self, dataset):
        pass

    def begin_epoch(self, dataset, t, epoch):
        pass

    def end_epoch(self, dataset, t, epoch):
        if self.scheduler is not None:
            self.scheduler.step()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass