import os
import copy
import time
from tqdm import trange
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
from scipy.ndimage.interpolation import rotate as scipyrotate

import wandb

from .base import Callback

from contlearn.utils.metrics import AccuracyPerClassMetric, TaskAccuracyMetric
from contlearn.utils.dc import DiffAugment, ParamDiffAug

class FintuneCallback(Callback):
    def __init__(self, args, model, dataset):
        super().__init__(args, model, dataset)
        self.args = args

    def on_train_end(self, model, train_loader_all, test_loader_all):
        if hasattr(model, "buffer"):
            buffer_data = model.buffer.get_all_data()
            image_ft, label_ft = buffer_data[0], buffer_data[1]

            print("The labels in the buffer used for finetuning:", 
                  torch.unique(label_ft.cpu(), return_counts=True))

            # The images are already normalized
            # Augmentation will be done in finetune_and_log function
            dst_train = TensorXYDataset(image_ft, label_ft, model.transform)
            train_loader = Data.DataLoader(dst_train, batch_size=train_loader_all.batch_size, shuffle=True, num_workers=0)

            # Finetuning the entire model
            net = copy.deepcopy(model.net)
            net.unfreeze()
            finetune_and_log("ft_real", net, train_loader, test_loader_all)

            # Linear Probing, finetuning only the classification layer
            net = copy.deepcopy(model.net)
            net.unfreeze()
            net.freeze_feature()
            # net.reinit_classifier()
            finetune_and_log("lp_real", net, train_loader, test_loader_all)
            
        else:
            print("FintuneCallback(): This model does not have a buffer")

def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

class TensorXYDataset(Data.TensorDataset):
    def __init__(self, images, labels, transform=None) -> None:
        assert images.size(0) == labels.size(0), "Size mismatch between tensors"
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        x = self.images[index]
        y = self.labels[index]
        
        if self.transform is not None:
            x = self.transform(x)

        return (x, y)

    def __len__(self):
        return self.images.size(0)

def finetune_and_log(name, model, train_loader, test_loader, n_epochs=200, lr_init=0.001):
    print("Start finetuning on", name)
    model, acc_train, acc_test, apc_test = finetune_dataset(model, train_loader, test_loader, wandb=wandb, name=name, n_epochs=n_epochs, lr_init=lr_init)

    if name is not None:
        wandb.log({
            "%s_train_acc" % name: acc_train,
            "%s_test_acc" % name: acc_test,
        })
        for i, acc in enumerate(apc_test):
            wandb.log({
                "class_id": i,
                "%s_apc" % name: acc
            })

    return model, acc_test

def finetune_dataset(model, train_loader, test_loader, wandb = None, name='ft', verbose=False, return_traj=False, n_epochs=None, device='cuda', lr_init=0.001):
    model = model.to(device)

    lr = lr_init
    Epoch = n_epochs

    # lr_schedule = [Epoch//2+1]
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    
    lr_schedule = [Epoch//2+1, Epoch//4*3+1]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(device)

    loss_test, acc_test, acc_per_class = epoch('test', test_loader, model, optimizer, criterion)
    # print('%s Evaluate_%02d: epoch = %04d, test acc = %.4f' % (get_time(), it_eval, 0,  acc_test))

    if return_traj:
        trajectory = []

    start = time.time()
    for ep in trange(Epoch+1):
        loss_train, acc_train = epoch('train', train_loader, model, optimizer, criterion)
        if ep in lr_schedule:
            # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
            
            lr *= 0.1
            # Set the learning rate of the optimizer to be new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        if ep % 20 == 0:
            loss_test, acc_test, acc_per_class = epoch('test', test_loader, model, optimizer, criterion)
            if verbose:
                time_train = time.time() - start
                print('%s Evaluate: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), ep, int(time_train), loss_train, acc_train, acc_test))
            if wandb is not None and name is not None:
                logs = {
                    "%s/epoch" % name: ep,
                    "%s/loss_train" % name: loss_train,
                    "%s/acc_train" % name: acc_train,
                }
                logs.update({
                    "%s/loss_test" % name: loss_test,
                    "%s/acc_test" % name: acc_test,
                })
                wandb.log(logs)

        if return_traj and ep % 10 == 0:
            trajectory.append([p.detach().cpu() for p in model.parameters()])

    time_train = time.time() - start
    loss_test, acc_test, acc_per_class = epoch('test', test_loader, model, optimizer, criterion)
    if verbose:
        print('%s Evaluate: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), Epoch, int(time_train), loss_train, acc_train, acc_test))
    
    if return_traj:
        return model, acc_train, acc_test, acc_per_class, trajectory
    else:
        return model, acc_train, acc_test, acc_per_class


def epoch(mode, dataloader, net, optimizer, criterion, device='cuda'):
    aug = (mode == "train")

    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(device)
    criterion = criterion.to(device)

    if mode == 'train':
        net.train()
    else:
        net.eval()
        apc_metric = AccuracyPerClassMetric(net.num_classes)

    for i_batch, batch in enumerate(dataloader):
        img = batch[0].float().to(device)

        if aug:
            img = DiffAugment(img, "color_crop_cutout_flip_scale_rotate", param=ParamDiffAug())

        lab = batch[1].long().to(device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # Compute per-classe accuracy
            y_pred = output.argmax(1)
            y = lab
            apc_metric.update(y, y_pred)

    loss_avg /= num_exp
    acc_avg /= num_exp

    if mode == 'train':
        return loss_avg, acc_avg
    else:
        acc_per_class = apc_metric.summary()
        return loss_avg, acc_avg, acc_per_class
