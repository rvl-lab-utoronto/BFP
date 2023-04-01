# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from syslog import LOG_PID
import wandb
from argparse import Namespace
import numpy as np

from utils.conf import base_path

class WandbLogger:

    def __init__(self, args: Namespace, setting: str, project: str = "mammoth", name = None) -> None:

        self.args = args
        self.project = project
        self.name = name

        # If it's class-il, also report task-il metrics. If it is task-il, no class-il to be reported
        self.settings = [setting]
        if setting == 'class-il':
            self.settings.append('task-il')

        # Record the accs over each step and report the avg inc acc at the end
        self.accs_cil_mean = []
        self.accs_til_mean = []

        # Record the acces over each step
        # outer index: at the end of training on task
        # inner index: the accuracy of task
        self.accs_cil = []
        self.accs_til = []

    def get_name(self) -> str:
        """
        :return: the name of the model
        """
        return self.name

    def log_task_end(self, accs_type_task):
        acc_cil_task, acc_til_task = accs_type_task

        # Record the accs at the end of training on the current task
        self.accs_cil.append(acc_cil_task)
        self.accs_til.append(acc_til_task)
        
        self.accs_cil_mean.append(np.mean(acc_cil_task))
        self.accs_til_mean.append(np.mean(acc_til_task))

        # log the accuracy and forgetting
        log_dict = {}
        log_dict['class-il/acc_avg_inc'] = np.mean(self.accs_cil_mean)
        log_dict['task-il/acc_avg_inc'] = np.mean(self.accs_til_mean)
        
        n_tasks = len(self.accs_cil)
        if n_tasks > 1:
            avg_forgetting_cil = 0
            avg_forgetting_til = 0
            for t in range(n_tasks - 1): # do not compute forgetting for the last task
                acc_cil_max = np.max([self.accs_cil[i][t] for i in range(t, n_tasks-1)])
                acc_til_max = np.max([self.accs_til[i][t] for i in range(t, n_tasks-1)])
                forget_cil = acc_cil_max - self.accs_cil[-1][t]
                forget_til = acc_til_max - self.accs_til[-1][t]
                avg_forgetting_cil += forget_cil
                avg_forgetting_til += forget_til
                log_dict[f'class-il-forget/task{t+1:02d}'] = forget_cil
                log_dict[f'task-il-forget/task{t+1:02d}'] = forget_til
            avg_forgetting_cil /= (n_tasks - 1)
            avg_forgetting_til /= (n_tasks - 1)
            log_dict['class-il-forget/avg'] = avg_forgetting_cil
            log_dict['task-il-forget/avg'] = avg_forgetting_til

        return log_dict

    def log_final(self):
        log_dict = {}
        
        n_tasks = len(self.accs_cil)
        avg_acc_cil = 0.0
        avg_acc_til = 0.0
        for t in range(n_tasks):
            # Log the immediate accuracies of each task after training on it
            learn_acc_cil = self.accs_cil[t][t]
            learn_acc_til = self.accs_til[t][t]
            avg_acc_cil += learn_acc_cil
            avg_acc_til += learn_acc_til
            log_dict[f'class-il-learn/acc_task{t+1:02d}'] = learn_acc_cil
            log_dict[f'task-il-learn/acc_task{t+1:02d}'] = learn_acc_til
        avg_acc_cil /= n_tasks
        avg_acc_til /= n_tasks
        log_dict['class-il-learn/acc_avg'] = avg_acc_cil
        log_dict['task-il-learn/acc_avg'] = avg_acc_til

        return log_dict

    def log_accuracy(self, all_accs: np.ndarray, all_mean_accs: np.ndarray,
                     type=None) -> None:
        """
        Logs the current accuracy value for each task.
        :param all_accs: the accuracies (class-il, task-il) for each task
        :param all_mean_accs: the mean accuracies for (class-il, task-il)
        :param args: the arguments of the run
        :param task_number: the task index
        """
        mean_acc_cil, mean_acc_til = all_mean_accs
        log_dict = {}

        if type == "final":
            log_dict.update(self.log_final())
        elif type == "end_task":
            log_dict.update(self.log_task_end(all_accs))

        for setting in self.settings:
            mean_acc = mean_acc_til if \
                setting == 'task-il' else mean_acc_cil
            index = 1 if setting == 'task-il' else 0
            accs = [all_accs[index][kk] for kk in range(len(all_accs[0]))]

            if type == "final":
                setting = setting + 'f'

            log_dict.update({
                f'{setting}/acc_task{kk + 1:02d}': acc
                for kk, acc in enumerate(accs)
            })
            log_dict[f'{setting}/acc_mean'] = mean_acc

        self.log_dict(log_dict)

    def log_loss(self, loss: float) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        :param args: the arguments of the run
        :param epoch: the epoch index
        :param task_number: the task index
        """
        log_dict = {}
        for setting in self.settings:
            log_dict[f'{setting}/loss'] = loss
        self.log_dict(log_dict)

    def log_loss_gcl(self, loss: float) -> None:
        """
        Logs the loss value at each iteration.
        :param loss: the loss value
        """
        log_dict = {}
        for setting in self.settings:
            log_dict[f'{setting}/loss'] = loss
        self.log_dict(log_dict)

    def log_activations(self, sample_act_map, avg_act_map, inst_sparsity,
                        dead_neuron_ratio):
        """Log activation results."""

        def _make_img(img):
            """[H, W] --> [H, W, 3] for logging."""
            return np.stack([img, img, img], axis=-1)

        # TODO: only log once?
        log_dict = {}
        log_dict['figures/avg_act_map'] = wandb.Image(_make_img(avg_act_map))
        for i, act_map in enumerate(sample_act_map):
            log_dict[f'figures/sample_act_map-{i}'] = wandb.Image(
                _make_img(act_map))
        log_dict['figures/inst_sparsity'] = inst_sparsity
        log_dict['figures/dead_neuron_ratio'] = dead_neuron_ratio
        self.log_dict(log_dict)

    def log_dict(self, log_dict):
        # append the name to each of the logged keys
        if self.name is not None and len(self.name) > 0:
            log_dict = {f'{self.name}-{k}': v for k, v in log_dict.items()}
        wandb.log(log_dict)

    def close(self) -> None:
        """
        At the end of the execution, closes the logger.
        """
        wandb.finish()
