# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')    

    parser.add_argument('--n_epochs', type=int,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')
    

    # parameters for backbone == 'convnet'
    parser.add_argument('--net_width', type=int, default=128,
                        help='Net width.')
    parser.add_argument('--net_depth', type=int, default=3,
                        help='Net depth.')
    parser.add_argument('--net_norm', type=str, default='batchnorm', 
                        choices=['batchnorm', 'layernorm', 'instancenorm', 'groupnorm', 'none'],
                        help='name of normalization layer, used for ConvNet')
    parser.add_argument('--net_pooling', type=str, default='maxpooling', 
                        choices=['maxpooling', 'avgpooling', 'none'],
                        help='name of pooling layer, used for ConvNet')

    parser.add_argument("--cum_loaders", action="store_true",
                        help="If set, the cumulative loaders will be used for training, like joint training oracle baseline. ")

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=0,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')
    parser.add_argument("--base_path", type=str, default='./data/', 
                        help="directory where logging and data will be stored. ")
    parser.add_argument("--ckpt_folder", type=str, default='./checkpoint/',
                        help="directory where checkpoints will be stored. ")

    parser.add_argument('--non_verbose', action='store_true')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')

    parser.add_argument("--project", type=str, default="mammoth", 
                        help="wandb project name")
    parser.add_argument('--n_runs', type=int, default=1, 
                        help="Number of times to repeat for this experiment")
    parser.add_argument('--exp_suffix', type=str, default=None, 
                        help="The suffix of the experiment name")
    parser.add_argument("--time_suffix", type=str, default=None,
                        help="If set, the given time suffix will be used. Otherwise, it will use the running time stamp. ")

    parser.add_argument("--save_weights", action="store_true", 
                        help="If set, the model weight be saved after each task")
    parser.add_argument("--load_backbone_weights", type=str, default=None,
                        help="If set, the backbone weights will be loaded from this file")
    parser.add_argument("--resume_path", type=str, default=None,
                        help="If set, the entire experiment will be resumed from this path")

    parser.add_argument("--classifier_type", type=str, default=None, choices=['linear', 'mlp-2'],
                        help="The type of the classifier")

    parser.add_argument("--freeze_feature", action="store_true",
                        help="If set, the feature extractor will be frozen during continual learning")
    parser.add_argument("--freeze_classifier", action="store_true",
                        help="If set, the classifier will be frozen (as randomly initialized) during continual learning")

    parser.add_argument("--preemptive", action="store_true",
                        help="If set, weight will be saved for training on preemptive cluster")

    parser.add_argument('--backbone', type=str, default=None, choices=['convnet', 'mnistmlp', 'resnet-in', 'resnet'],
                        help='Name of the backbone network. if not set, default of each dataset will be used. ')
    parser.add_argument("--backbone_norm", type=str, default='batchnorm', 
                        choices=['batchnorm', 'layernorm', 'instancenorm', 'groupnorm', 'none'],
                        help="name of normalization layer, used for backbone")


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True, 
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, 
                        help='The batch size of the memory buffer.')
    parser.add_argument("--class_balance", type=str2bool, default=True,
                        help="If set, the memory buffer will be balanced by class")
