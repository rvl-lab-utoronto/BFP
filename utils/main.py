# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import numpy # needed (don't change it)
import importlib
import os
import sys
import socket
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')

from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import ContinualDataset
from datasets import get_dataset
from models import get_model
from utils.training import run_epoch, evaluate, print_mean_accuracy
from utils.best_args import best_args
from utils.conf import set_random_seed, base_path, set_base_path
from utils.callbacks import FintuneCallback, CheckpointCallback
from utils.wandb_logger import WandbLogger
from backbone import get_classifier
from backbone.MNISTMLP import MNISTMLP
from backbone.ConvNet import ConvNet

import numpy as np
import torch
import uuid
import datetime
import wandb


time_suffix = datetime.datetime.now().strftime('%y%m%d%H%M%S')

def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

def parse_args(to_parse=None):
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_known_args(to_parse)[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')

        # Up to this point, the parsed arguments are only basic ones and used for get the best args
        args = parser.parse_known_args(to_parse)[0]

        # get the best args for the given model, dataset and buffer size
        best = best_args[args.dataset][args.model]
        
        if hasattr(mod, 'Buffer'):
            best = best[args.buffer_size]
        else:
            best = best[-1]

        # Get the parser from the specific model and dataset
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()

        # Modified the input arguments according to best_args
        if to_parse is None:
            to_parse = sys.argv[1:]

        to_parse = to_parse + ['--' + k + '=' + str(v) 
                               for k, v in best.items() 
                               if '--' + k not in to_parse]
                               
        to_parse.remove('--load_best_args')
        
        # Parse the arguments
        args = parser.parse_args(to_parse)
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args(to_parse)

    if args.model == 'joint' and args.dataset == 'mnist-360':
        args.model = 'joint_gcl'

    if args.seed is not None:
        set_random_seed(args.seed)

    exp_name = f'{args.model}-{args.dataset}'
    if 'buffer_size' in vars(args).keys():
        exp_name += f'-buf_{args.buffer_size}'
    if args.exp_suffix is not None:
        exp_name += f'-{args.exp_suffix}'

    if args.time_suffix is not None:
        exp_name += "-" + args.time_suffix
    else:
        exp_name += "-" + time_suffix
        
    args.exp_name = exp_name
    
    return args

def process_args(args):
    # Set environment variables for ComputeCanada
    if "SLURM_TMPDIR" in os.environ:
        args.base_path = os.path.join(os.environ["SLURM_TMPDIR"], 'data/')
        args.ckpt_folder = os.path.join(os.environ["SLURM_TMPDIR"], 'ckpt/')

    set_base_path(args.base_path)

    return args

def get_backbone(args, dataset):
    if args.backbone == 'convnet':
        im_size = get_dataset(args).IMG_SIZE
        channel = get_dataset(args).N_CHANNELS
        num_classes = get_dataset(args).N_CLASSES
        backbone = ConvNet(channel=channel, 
                        num_classes=num_classes, 
                        net_width=args.net_width, 
                        net_depth=args.net_depth, 
                        net_act='relu', 
                        net_norm=args.net_norm, 
                        net_pooling=args.net_pooling,
                        im_size=im_size, 
                        )

    elif args.backbone == 'mnistmlp':
        backbone = MNISTMLP(28 * 28, 10)
    else:
        backbone = dataset.get_backbone(args)

    # Change the classification layer of the backbone according to the arguments
    if args.classifier_type is not None:
        backbone.classifier = get_classifier(args.classifier_type, backbone.feat_dim, backbone.num_classes)

    # Used for loading pretrained models
    if args.load_backbone_weights is not None:
        print("Loading weights from", args.load_backbone_weights)
        ckpt = torch.load(args.load_backbone_weights)
        backbone.load_state_dict(ckpt['state_dict'])

    print("Using backbone", type(backbone).__name__)

    return backbone

def create_or_restore_training_state(args, ckpt_path=None):
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        args.exp_name = ckpt['args'].exp_name
    else:
        ckpt = None

    # Build dataset class (for meta data only)
    dataset = get_dataset(args)

    # Build backbone network and continual model
    backbone = get_backbone(args, dataset)
    model = get_model(args, backbone, dataset.get_loss(), dataset.get_transform())
    model.reset_optimizers(dataset)

    # Set some arguments to default ones if still not set at this point
    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
        assert args.n_epochs is not None, 'Could not get number of epochs'
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
        assert args.batch_size is not None, 'Could not get batch size'
    if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and (not hasattr(args, "minibatch_size") or args.minibatch_size is None):
        args.minibatch_size = dataset.get_minibatch_size()
        assert args.minibatch_size is not None, 'Could not get minibatch size'

    # Re-create the dataset to reflect some argument changes and build dataloaders
    dataset = get_dataset(args)
    dataset.build_data_loaders()
    if args.cum_loaders:
        dataset.build_data_loaders_up_to()

    # Initialize logger and callbacks
    # Always using wandb for the logger
    logger = WandbLogger(args, dataset.SETTING, args.project)

    # build callbacks
    callbacks = []
    if args.save_weights:
        callbacks.append(CheckpointCallback(args, model, dataset))

    # Finetune callback is always used
    callbacks.append(FintuneCallback(args, model, dataset))

    if ckpt is not None:
        ckpt = torch.load(ckpt_path)
        
        if hasattr(model, "buffer"):
            # initialize the variables in buffer
            for attr_str in model.buffer.attributes:
                key = "buffer." + attr_str
                if key in ckpt['state_dict']:
                    model.buffer.register_buffer(attr_str, ckpt['state_dict'][key])

        if "old_net.classifier.weight" in ckpt['state_dict']:
            model.old_net = copy.deepcopy(model.net)

        # Load model
        model.load_state_dict(ckpt['state_dict'])
        model.opt.load_state_dict(ckpt['opt_state_dict'])
        if model.scheduler is not None:
            model.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        start_run = ckpt['run']
        start_task = ckpt['task']
        start_epoch = ckpt['epoch']
        wandb_id = ckpt['wandb_id']

        wandb.init(
            project=args.project, 
            id=wandb_id,
            resume="must",
            dir=args.ckpt_folder,
            settings=wandb.Settings(start_method="fork")
        )
        print("Experiment resumed from", ckpt_path)
    else:
        wandb_id = wandb.util.generate_id()
        wandb.init(
            project=args.project, 
            name=args.exp_name, 
            id=wandb_id,
            config=args,
            dir=args.ckpt_folder,
            reinit=True,
            settings=wandb.Settings(start_method="fork")
        )
        start_run = 0
        start_task = 0
        start_epoch = 0

    return model, dataset, logger, callbacks, start_run, start_task, start_epoch, wandb_id

def save_training_state(args, model, dataset, run_id, task_id, epoch, wandb_id):
    temp_path = os.path.join(args.ckpt_folder, "temp.ckpt")
    ckpt_path = os.path.join(args.ckpt_folder, "latest.ckpt")

    state_dict = model.state_dict()

    ckpt = {
        'state_dict': state_dict,
        "opt_state_dict": model.opt.state_dict(),
        'run': run_id,
        'task': task_id,
        'epoch': epoch,
        'wandb_id': wandb_id,
        'args': args,
    }

    if model.scheduler is not None:
        ckpt["scheduler_state_dict"] = model.scheduler.state_dict()

    torch.save(ckpt, open(temp_path, 'wb'))

    os.replace(temp_path, ckpt_path)

def main(args):
    # whether to resume training or start a new one
    if args.preemptive:
        if args.resume_path is None:
            args.resume_path = os.path.join(args.ckpt_folder, "latest.ckpt")
        if not os.path.exists(args.resume_path):
            args.resume_path = None

    model, dataset, logger, callbacks, start_run, start_task, start_epoch, wandb_id = \
        create_or_restore_training_state(args, args.resume_path)

    print("Starting training from run", start_run, "task", start_task, "epoch", start_epoch)

    # If not running on a preemptive cluster, append the ckpt_folder with the experiment name
    if not args.preemptive:
        args.ckpt_folder = os.path.join(args.ckpt_folder, args.exp_name)
        
    if not os.path.exists(args.ckpt_folder):
        os.makedirs(args.ckpt_folder)
        
    train_loader_all = dataset.train_loader_all
    test_loader_all = dataset.test_loader_all

    for run_id in range(start_run, args.n_runs):
        set_random_seed(args.seed + run_id)
        if start_task == 0 and start_epoch == 0  and run_id > start_run:
            # Reset the state at the beginning of each run
            model, dataset, logger, callbacks, _, start_task, start_epoch, wandb_id = \
                create_or_restore_training_state(args, None)

        # Move the model to the proper device
        model.net.to(model.device)
        if hasattr(model, "old_net") and model.old_net is not None:
            model.old_net.to(model.device)
            
        for task_id in range(start_task, dataset.N_TASKS):
            model.net.train()
            if args.cum_loaders:
                train_loader, _ = dataset.get_data_loaders_up_to_task(task_id)
            else:
                train_loader, _ = dataset.get_data_loaders_by_task(task_id)
            model.begin_task(dataset, task_id, start_epoch)

            for epoch in range(start_epoch, model.args.n_epochs):
                # Save checkpoint at the beginning of each epoch
                if args.preemptive:
                    save_training_state(args, model, dataset, run_id, task_id, epoch, wandb_id)

                model.begin_epoch(dataset, task_id, epoch)

                run_epoch(model, train_loader, logger, callbacks, task_id, epoch)
                
                model.end_epoch(dataset, task_id, epoch)
                
                # callbacks for epoch end
                for callback in callbacks:
                    callback.on_train_epoch_end(
                        epoch=epoch, 
                        model=model,
                        task_train_loader=train_loader,
                    )

                # Evaluate the model every a few epochs
                if epoch % 5 == 0:
                    accs, mean_acc = evaluate(model, dataset, task_id=task_id)
                    logger.log_accuracy(np.array(accs), mean_acc)
            
            # At the end of a task
            if hasattr(model, 'end_task'):
                model.end_task(dataset)

            accs, mean_acc = evaluate(model, dataset, task_id=task_id)
            print_mean_accuracy(mean_acc, task_id + 1, dataset.SETTING)
            logger.log_accuracy(np.array(accs), mean_acc, type="end_task")

            for callback in callbacks:
                callback.on_task_end(
                    task_id=task_id, 
                    model=model,
                    task_train_loader=train_loader,
                )
            
            start_epoch = 0
    
        # At the end of the experiment
        # Log the final accuracy
        logger.log_accuracy(np.array(accs), mean_acc, type="final")

        for callback in callbacks:
            callback.on_train_end(
                model=model, 
                train_loader_all=train_loader_all, 
                test_loader_all=test_loader_all
            )

        logger.close()

        start_task = 0

if __name__ == '__main__':
    lecun_fix()
    args = parse_args()
    args = process_args(args)
    main(args)
