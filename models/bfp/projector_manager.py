# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.nn import functional as F

from utils.args import *
from utils.routines import forward_loader_all_layers

from .utils import pool_feat, match_loss

def add_parser(parser):
	parser.add_argument('--alpha_bfp', type=float, required=True,
				help="Weight of the backward feature projection loss. It can be overridden by the 'alpha_bfpX' below")
	parser.add_argument('--alpha_bfp1', type=float, default=None)
	parser.add_argument('--alpha_bfp2', type=float, default=None)
	parser.add_argument('--alpha_bfp3', type=float, default=None)
	parser.add_argument('--alpha_bfp4', type=float, default=None)

	parser.add_argument('--loss_type', type=str, default='mfro', choices=['mse', 'rmse', 'mfro', 'cos'],
				help='How to compute the matching loss on projected features.')
	parser.add_argument("--normalize_feat", action="store_true",
				help="if set, normalize features before computing the matching loss.")
	parser.add_argument("--opt_type", type=str, default="sgdm", choices=["sgd", "sgdm", "adam"],
				help="Optimizer type.")
	parser.add_argument("--proj_lr", type=float, default=0.1,
				help="Learning rate for the optimizer on the projectors.")    
	parser.add_argument("--momentum", type=float, default=0.9,
				help="Momentum for SGD.")

	parser.add_argument('--proj_init_identity', action="store_true",
				help="If set, initialize the projectors to the identity mapping.")
	parser.add_argument('--proj_task_reset', type=str2bool, default=True,
				help="If set, initialize the projectors to a random mapping.")

	parser.add_argument('--proj_type', type=str, default="1", choices=['0', '1', '2', '0p+1'],
				help="Type of the backward feature projection. (number of layers in MLP projector)")
	parser.add_argument('--final_feat', action='store_true',
				help="If true, bfp loss will only be applied to the last feature map.")
	parser.add_argument('--pool_dim', default='hw', type=str, choices=['h', 'w', 'c', 'hw', 'flatten'], 
				help="Pooling before computing BFP loss. If None, no pooling is applied.")			
	
	return parser

class ProjectorManager(nn.Module):
	'''
	Helper class managing the projection layers for BFP
	Such that it can be easily integrated into other continual learning methods
	'''
	def __init__(self, args, net_channels, device):
		super(ProjectorManager, self).__init__()
		self.args = args
		self.net_channels = net_channels
		self.device = device
		
		# Initialize the backward projection layers
		self.alpha_bfp_list = [self.args.alpha_bfp] * len(self.net_channels)
		if self.args.alpha_bfp1 is not None: self.alpha_bfp_list[0] = self.args.alpha_bfp1
		if self.args.alpha_bfp2 is not None: self.alpha_bfp_list[1] = self.args.alpha_bfp2
		if self.args.alpha_bfp3 is not None: self.alpha_bfp_list[2] = self.args.alpha_bfp3
		if self.args.alpha_bfp4 is not None: self.alpha_bfp_list[3] = self.args.alpha_bfp4
		self.bfp_flag = sum(self.alpha_bfp_list) > 0
		
		self.reset_proj()
		
		# Get the list of layers where BFP is applied
		if self.args.final_feat:
			self.layers_bfp = [-1]
		else:
			self.layers_bfp = list(range(len(self.net_channels)))

	def begin_task(self, dataset, t=0, start_epoch=0):
		self.task_id = t
		if not self.bfp_flag: return

		if self.args.proj_task_reset:
			self.reset_proj()

	def _get_projector(self, feat_dim, init_identity=False):
		if self.args.proj_type == '0':
			projector = nn.Identity()
		elif self.args.proj_type == "1":
			projector = nn.Linear(feat_dim, feat_dim)
			if init_identity:
				projector.weight.data = torch.eye(feat_dim)
				projector.bias.data = torch.zeros(feat_dim)
		elif self.args.proj_type == "2":
			projector = nn.Sequential(
				nn.Linear(feat_dim, feat_dim),
				nn.ReLU(),
				nn.Linear(feat_dim, feat_dim),
			)
		else:
			raise Exception("Unknown projector type: {}".format(self.args.proj_type))

		projector.to(self.device)
		return projector
		
	def reset_proj(self):
		# Get one optimizer for each network layer
		self.projectors = nn.ModuleList()
		for c in self.net_channels:
			projector = self._get_projector(c, self.args.proj_init_identity)
			self.projectors.append(projector)

		if self.args.proj_type != '0':
			# Optimizer for all projectors
			if self.args.opt_type == 'sgd':
				self.opt_proj = SGD(
					sum([list(p.parameters()) for p in self.projectors], []), 
					lr=self.args.proj_lr)
			elif self.args.opt_type == 'sgdm':
				self.opt_proj = SGD(
					sum([list(p.parameters()) for p in self.projectors], []), 
					lr=self.args.proj_lr, momentum=self.args.momentum)
			elif self.args.opt_type == 'adam':
				self.opt_proj = Adam(
					sum([list(p.parameters()) for p in self.projectors], []), 
					lr=self.args.proj_lr)
		else:
			self.opt_proj = None

	def compute_loss(self, feats, feats_old, mask_new, mask_old):
		bfp_loss = 0.0

		for i in self.layers_bfp:
			projector = self.projectors[i]
			feat = feats[i]
			feat_old = feats_old[i]
			
			# After pooling, feat and feat_old have shape (n, d)
			feat, feat_old = pool_feat(feat, feat_old, self.args.pool_dim, self.args.normalize_feat)
			
			feat_proj = projector(feat) # (N, C)
			
			bfp_loss += self.alpha_bfp_list[i] * match_loss(feat_proj, feat_old, self.args.loss_type)

		bfp_loss /= len(self.layers_bfp)

		loss = bfp_loss

		loss_dict = {
			'match_loss': bfp_loss,
		}

		return loss, loss_dict

	def before_backward(self):
		if not self.bfp_flag: return
		if self.opt_proj is not None: self.opt_proj.zero_grad()

	def end_task(self, dataset, net):
		if not self.bfp_flag: return

	def step(self):
		if not self.bfp_flag: return
		if self.opt_proj is not None: self.opt_proj.step()
		