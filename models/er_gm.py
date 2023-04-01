# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import wandb
import torch
import torch.nn.functional as F


from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel

from contlearn.utils.loss import LGMLoss

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Experience Replay with Gaussian Mixture loss.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Margin parameter for the GM loss.")
    parser.add_argument("--lam", type=float, default=0.01,
                        help="Weight of the regulaization term in the GM loss.")
    parser.add_argument("--gm_lr", type=float, default=1e-3,
                        help="Learning rate for the GM loss.")
    return parser


class ErGm(ContinualModel):
    NAME = 'er_gm'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErGm, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.gm_loss = LGMLoss(backbone.num_classes, backbone.feat_dim, self.args.alpha, self.args.lam)
        self.gm_loss.to(self.device)
        self.opt_gm = torch.optim.SGD(self.gm_loss.parameters(), lr=self.args.gm_lr, momentum=0.5)

        self.net.add_gm_loss(self.gm_loss)

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        self.opt_gm.zero_grad()

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        feats = self.net(inputs, returnt="features")
        logits, reg_loss = self.gm_loss(feats, labels)
        ce_loss = F.cross_entropy(logits, labels)
        loss = ce_loss + reg_loss
        
        loss.backward()
        self.opt.step()
        self.opt_gm.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        # centers = self.gm_loss.means
        # print(torch.cdist(centers, centers))

        wandb.log({
            "train/loss": loss, 
            "train/ce_loss": ce_loss, 
            "train/reg_loss": reg_loss,
        })

        return loss.item()
