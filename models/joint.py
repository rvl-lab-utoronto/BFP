# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim import SGD

from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets.utils.validation import ValidationDataset
from utils.status import progress_bar
import torch
import numpy as np
import math
from torchvision import transforms

from tqdm import tqdm


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    parser.add_argument('--fitting_epochs', type=int, default=50,
                        help='Penalty weight.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class Joint(ContinualModel):
    NAME = 'joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Joint, self).__init__(backbone, loss, args, transform)
        self.old_data = []
        self.old_labels = []

    def begin_task(self, dataset, t=0, start_epoch=0):
        super().begin_task(dataset, t, start_epoch)

    def end_task(self, dataset):
        train_loader = dataset.train_loaders[self.task_id]

        if dataset.SETTING != 'domain-il':
            self.old_data.append(train_loader.dataset.data)
            self.old_labels.append(torch.tensor(train_loader.dataset.targets))

            # # for non-incremental joint training
            if self.task_id < dataset.N_TASKS - 1: return

            # reinit network
            self.net = dataset.get_backbone(self.args)
            self.net.to(self.device)
            self.net.train()
            self.opt = SGD(self.net.parameters(), lr=self.args.lr)

            # prepare dataloader
            all_data, all_labels = None, None
            for i in range(len(self.old_data)):
                if all_data is None:
                    all_data = self.old_data[i]
                    all_labels = self.old_labels[i]
                else:
                    all_data = np.concatenate([all_data, self.old_data[i]])
                    all_labels = np.concatenate([all_labels, self.old_labels[i]])

            transform = dataset.TRANSFORM if dataset.TRANSFORM is not None else transforms.ToTensor()

            temp_dataset = ValidationDataset(all_data, all_labels, transform=transform)
            loader = torch.utils.data.DataLoader(temp_dataset, batch_size=self.args.batch_size, shuffle=True)

            # train
            for e in range(self.args.fitting_epochs):
                for i, batch in tqdm(enumerate(loader), total=len(loader), leave=False, desc='Joint ep {}'.format(e)):
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()
                    self.opt.step()
        else:
            self.old_data.append(train_loader)
            # train
            if self.task_id < dataset.N_TASKS - 1: return
            
            all_inputs = []
            all_labels = []
            for source in self.old_data:
                for x, l, _ in source:
                    all_inputs.append(x)
                    all_labels.append(l)
            all_inputs = torch.cat(all_inputs)
            all_labels = torch.cat(all_labels)
            bs = self.args.batch_size
            scheduler = dataset.get_scheduler(self, self.args)

            for e in range(self.args.fitting_epochs):
                order = torch.randperm(len(all_inputs))
                for i in range(int(math.ceil(len(all_inputs) / bs))):
                    inputs = all_inputs[order][i * bs: (i+1) * bs]
                    labels = all_labels[order][i * bs: (i+1) * bs]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()
                    self.opt.step()
                    progress_bar(i, int(math.ceil(len(all_inputs) / bs)), e, 'J', loss.item())
                
                if scheduler is not None:
                    scheduler.step()

    def observe(self, inputs, labels, not_aug_inputs):
        return 0
