# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara).
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data as Data
from utils.wandb_logger import WandbLogger
from utils.callbacks import *
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
import sys

def print_mean_accuracy(mean_acc: np.ndarray, task_number: int,
                        setting: str) -> None:
    """
    Prints the mean accuracy on stderr.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    if setting == 'domain-il':
        mean_acc, _ = mean_acc
        print('\nAccuracy for {} task(s): {} %'.format(
            task_number, round(mean_acc, 2)), file=sys.stderr)
    else:
        mean_acc_class_il, mean_acc_task_il = mean_acc
        print('\nAccuracy for {} task(s): \t [Class-IL]: {} %'
              ' \t [Task-IL]: {} %\n'.format(task_number, round(
            mean_acc_class_il, 2), round(mean_acc_task_il, 2)), file=sys.stderr)

def evaluate(model: ContinualModel, dataset: ContinualDataset, task_id:int, net=None) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :param task_id: The id of the current task (evaluates up to this task)
    :return: two tuples containing (cil_accs, til_accs), (cil_acc_alltask, til_acc_alltask)
    """
    if net is None:
        net = model.net
    
    status = net.training
    net.eval()
    accs, accs_mask_classes = [], []
    correct_alltask, correct_alltask_mask_classes, total_alltask = 0.0, 0.0, 0.0
    for t in range(task_id + 1):
        test_loader = dataset.test_loaders[t]
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0

        # Avoid division by zero
        total += 1e-7
        
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = net(inputs, t)
                else:
                    outputs = net(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    dataset.mask_classes(outputs, t)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

        correct_alltask += correct
        correct_alltask_mask_classes += correct_mask_classes
        total_alltask += total
    
    cil_acc_alltask = correct_alltask / total_alltask * 100
    til_acc_alltask = correct_alltask_mask_classes / total_alltask * 100

    net.train(status)

    return (accs, accs_mask_classes), (cil_acc_alltask, til_acc_alltask)

def run_epoch(model, train_loader, logger, callbacks, t, epoch) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param train_loader: the dataloader for the training set
    :param args: the arguments of the current execution
    """

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, desc='Task {},Ep {}'.format(t, epoch)):
        data = [data_item.to(model.device) for data_item in data]
        if hasattr(train_loader.dataset, 'logits'):
            inputs, labels, not_aug_inputs, logits = data
            loss = model.observe(inputs, labels, not_aug_inputs, logits)
        else:
            inputs, labels, not_aug_inputs = data
            loss = model.observe(inputs, labels, not_aug_inputs)

        logger.log_loss(loss)

        for callback in callbacks:
            callback.on_train_batch_end(
                model=model,
                batch=data,
            )
    