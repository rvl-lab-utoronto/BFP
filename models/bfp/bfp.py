# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import torch
import wandb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.nn import functional as F

from utils.buffer import Buffer
from models.utils.continual_model import ContinualModel
from backbone.MNISTMLP import MNISTMLP
from utils.args import *
from utils.lowrank_reg import LowRankReg
from utils.routines import forward_loader_all_layers

from .args import set_best_args
from .utils import *
from .projector_manager import ProjectorManager

class Bfp(ContinualModel):
    NAME = 'bfp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        set_best_args(args)

        # Then call the super constructor
        super(Bfp, self).__init__(backbone, loss, args, transform)
        self.old_net = None
        self.buffer = Buffer(self.args.buffer_size, self.device, class_balance = self.args.class_balance)

        # if resnet_skip_relu, modify the backbone to skip relu at the end of each major block
        if self.args.resnet_skip_relu:
            self.net.skip_relu(last=self.args.final_feat)

        # For domain-IL MNIST datasets, we should use the logits from the buffer
        if self.args.dataset in ['perm-mnist', 'rot-mnist']:
            self.args.use_buf_logits = True

        # if the old net is not used, then set the old_only and use_buf_logits flags
        if self.args.no_old_net:
            self.args.old_only = True
            self.args.use_buf_logits = True

        assert not (self.args.new_only and self.args.old_only)

        # initialize the projectors used for BFP
        self.projector_manager = ProjectorManager(self.args, self.net.net_channels, self.device)

    def begin_task(self, dataset, t=0, start_epoch=0):
        super().begin_task(dataset, t, start_epoch)
        self.projector_manager.begin_task(dataset, t, start_epoch)

    def observe(self, inputs, labels, not_aug_inputs):
        # Regular CE loss on the online data
        outputs, feats = self.net.forward_all_layers(inputs)
        ce_loss = self.loss(outputs, labels)

        def sample_buffer_and_forward(transform = self.transform):
            buf_data = self.buffer.get_data(self.args.minibatch_size, transform=transform)
            buf_inputs, buf_labels, buf_logits, buf_task_labels = buf_data[0], buf_data[1], buf_data[2], buf_data[3]
            buf_feats = [buf_data[4]] if self.args.use_buf_feats else None
            buf_logits_new_net, buf_feats_new_net = self.net.forward_all_layers(buf_inputs)

            return buf_inputs, buf_labels, buf_logits, buf_task_labels, buf_feats, buf_logits_new_net, buf_feats_new_net

        logits_distill_loss = 0.0
        replay_ce_loss = 0.0
        bfp_loss_all = 0.0
        bfp_loss_dict = None

        if not self.buffer.is_empty():
            '''Distill loss on the replayed images'''
            if self.args.alpha_distill > 0:
                if self.args.no_resample and "buf_inputs" in locals(): pass # No need to resample
                else: buf_inputs, buf_labels, buf_logits, buf_task_labels, buf_feats, buf_logits_new_net, buf_feats_new_net = sample_buffer_and_forward()

                if (not self.args.use_buf_logits) and (self.old_net is not None):
                    with torch.no_grad():
                        buf_logits = self.old_net(buf_inputs)
                        
                logits_distill_loss = self.args.alpha_distill * F.mse_loss(buf_logits_new_net, buf_logits)

            '''CE loss on the replayed images'''
            if self.args.alpha_ce > 0:
                if self.args.no_resample and "buf_inputs" in locals(): pass # No need to resample
                else: buf_inputs, buf_labels, buf_logits, buf_task_labels, buf_feats, buf_logits_new_net, buf_feats_new_net = sample_buffer_and_forward()
                
                replay_ce_loss = self.args.alpha_ce * self.loss(buf_logits_new_net, buf_labels)

            '''Backward feature projection loss'''
            if self.old_net is not None and self.projector_manager.bfp_flag:
                if not self.args.new_only:
                    buf_inputs, buf_labels, buf_logits, buf_task_labels, buf_feats, buf_logits_new_net, buf_feats_new_net = sample_buffer_and_forward()

                if self.args.use_buf_feats:
                    # new and old features should be both a list 
                    # And in this case, we only care about the last layer
                    feats_comb = buf_feats_new_net[-1:]
                    feats_old = buf_feats
                    mask_new = torch.ones_like(buf_labels).bool()
                    mask_old = torch.zeros_like(buf_labels).bool()
                else:
                    # Inputs, feats and labels for the online and buffer data, concatenated
                    if self.args.new_only:
                        inputs_comb = inputs
                        labels_comb = labels
                        feats_comb = feats
                    elif self.args.old_only:
                        mask_old = buf_labels < self.task_id * self.args.N_CLASSES_PER_TASK
                        inputs_comb = buf_inputs[mask_old]
                        labels_comb = buf_labels[mask_old]
                        feats_comb = [f[mask_old] for f in  buf_feats_new_net]
                    else:
                        inputs_comb = torch.cat((inputs, buf_inputs), dim=0)
                        labels_comb = torch.cat((labels, buf_labels), dim=0)
                        feats_comb = [torch.cat((f, bf), dim=0) for f, bf in zip(feats, buf_feats_new_net)]

                    mask_old = labels_comb < self.task_id * self.args.N_CLASSES_PER_TASK
                    mask_new = labels_comb >= self.task_id * self.args.N_CLASSES_PER_TASK

                    # Forward data through the old network to get the old features
                    with torch.no_grad():
                        self.old_net.eval()
                        _, feats_old = self.old_net.forward_all_layers(inputs_comb)
                
                bfp_loss_all, bfp_loss_dict = self.projector_manager.compute_loss(
                    feats_comb, feats_old, mask_new, mask_old)
                
        loss = ce_loss + logits_distill_loss + replay_ce_loss + bfp_loss_all

        self.opt.zero_grad()
        self.projector_manager.before_backward()

        loss.backward()
        
        self.opt.step()
        self.projector_manager.step()

        task_labels = torch.ones_like(labels) * self.task_id
        if self.args.use_buf_feats:
            # Store the unpooled version of the final-layer features in the buffer
            final_feats = feats[-1]
            self.buffer.add_data(examples=not_aug_inputs,
                                labels=labels,
                                logits=outputs.data, 
                                task_labels=task_labels,
                                final_feats=final_feats.data)
        else:
            self.buffer.add_data(examples=not_aug_inputs,
                                labels=labels,
                                logits=outputs.data, 
                                task_labels=task_labels)

        log_dict = {
            "train/loss": loss, 
            "train/ce_loss": ce_loss, 
            "train/logits_distill_loss": logits_distill_loss,
            "train/replay_ce_loss": replay_ce_loss,
            "train/bfp_loss_all": bfp_loss_all,
        }
        if bfp_loss_dict is not None:
            for k, v in bfp_loss_dict.items(): log_dict.update({"train/" + k: v})
        wandb.log(log_dict)

        return loss.item()

    def end_task(self, dataset):
        self.old_net = copy.deepcopy(self.net)
        self.old_net.eval()

        self.projector_manager.end_task(dataset, self.old_net)