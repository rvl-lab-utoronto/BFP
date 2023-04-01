import os, sys, copy

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)

import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset

from utils.helpers import load_models
from utils.callbacks.finetune import finetune_and_log
from utils.main import get_backbone
import wandb

# Get the dataset and dataloaders by loading the first model
def update_args(args):
    args.decompose_proj = False
    args.base_method = "derpp"
    args.svd_sep = False
    args.proj_feat_svd = False
    args.new_only = False
    args.no_old_net = False
    args.class_balance = True

def get_model_dataset(ckpt_folder, build_loader_up_to=False):
    models, dataset, args = load_models(ckpt_folder, n_models=1, update_args=update_args)

    dataset.build_data_loaders()
    train_loaders = dataset.train_loaders
    test_loaders = dataset.test_loaders

    train_loader_all = dataset.train_loader_all
    test_loader_all = dataset.test_loader_all

    if build_loader_up_to:
        dataset.build_data_loaders_up_to()
        train_loaders_up_to = dataset.train_loaders_up_to
        test_loaders_up_to = dataset.test_loaders_up_to

    # Load all the models
    n_classes = dataset.N_CLASSES
    n_tasks = dataset.N_TASKS
    classes_per_task = dataset.N_CLASSES_PER_TASK
    models, _ , _ = load_models(ckpt_folder, n_models=n_tasks, update_args=update_args)

    return dataset, models

def study_linear_prob(model, train_dataset, test_dataset, name, wandb_log = False, portions = [0.01, 0.1/3, 0.1, 1.0/3, 1.0]):
    # # Get the full training and testing dataset of all the tasks
    # train_dataset, test_dataset = dataset.get_datasets()

    # Get the dataloader for test data
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # Get the linear probing results using different amount of data
    results = []
    for p in portions:
        n_train = int(len(train_dataset) * p)
        print("Portion: ", p, "n_train: ", n_train)

        # Get a subset of the training data with a random portion of each class
        idx = np.random.permutation(len(train_dataset))[:n_train]
        train_subset = Subset(train_dataset, idx)
        train_loader = DataLoader(train_subset, batch_size=128, shuffle=False, num_workers=4)
        
        if wandb_log:
            wandb.init(
                project="mammoth_linear_prob",
                name=name + "_p" + str(p),
                config={
                    "portion": p,
                    "n_train": n_train,
                },
                reinit=True,
            )

        finetuned_model = copy.deepcopy(model)
        finetuned_model.freeze_feature()
        finetuned_model.reinit_classifier()
        finetuned_model, acc_test = finetune_and_log(
            "lp" if wandb_log else None, finetuned_model, train_loader, test_loader, n_epochs=200, lr_init=0.03)
        print("Test accuracy: ", acc_test)
        
        results.append({
            "portion": p,
            "n_train": n_train,
            "test_acc": acc_test, 
            "name": name
        })

    return results

if __name__ == "__main__":
    # Get the first arguments from the command line as the dataset name
    dataset_name = sys.argv[1]

    if dataset_name == "cifar10":
        ckpt_folders = [
            ("CIFAR10 SGD", "/home/qiao/src/mammoth/checkpoint/weights/sgd-seq-cifar10-221027205958",),
            ("CIFAR10 DER++", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar10-buf_200-derpp_bfp0_v2_ep50-220918165729",),
            ("CIFAR10 DER++ BFP L1", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar10-buf_200-derpp_l1_bfp1_hw_v2_ep50-221005213025",),
            ("CIFAR10 SGD BFP L1", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar10-buf_200-sgd_l1_bfp1_fin_hw_noreplay-221027012105"),
            ("CIFAR10 cumulative train", "/home/qiao/src/mammoth/checkpoint/weights/joint-seq-cifar10-221027224802"),
        ]
    elif dataset_name == "cifar100":
        ckpt_folders = [
            ("CIFAR100 SGD", "/home/qiao/src/mammoth/checkpoint/weights/sgd-seq-cifar100-221020221356"),
            ("CIFAR100 DER++", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar100-buf_500-derpp_bfp0_v2_ep50-220919002820",),
            ("CIFAR100 DER++ BFP L1", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar100-buf_500-derpp_l1_bfp1_hw_v2_ep50-221005213029",),
            ("CIFAR100 SGD BFP L1", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar100-buf_500-sgd_l1_bfp1_fin_hw_noreplay-221027012112"),
            ("CIFAR100 cumulative train", "/home/qiao/src/mammoth/checkpoint/weights/sgd-seq-cifar100-cum_train-221021000620"),
        ]
    elif dataset_name == "cifar10_m1":
        ckpt_folders = [
            ("CIFAR10 DER++ BFP L0", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar10-buf_200-derpp_l0_bfp1_hw_v2_ep50-221006160648",),
            ("CIFAR10 SGD BFP L0", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar10-buf_200-sgd_l0_bfp1_fin_hw_noreplay-221026213638"),
            ("CIFAR10 SGD BFP L2", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar10-buf_200-sgd_l2_bfp1_fin_hw_noreplay-221027045147"),
        ]
    elif dataset_name == "cifar100_m1":
        ckpt_folders = [
            ("CIFAR100 DER++ BFP L0", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar100-buf_500-derpp_l0_bfp1_hw_v2_ep50-221006160704",),
            ("CIFAR100 SGD BFP L0", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar100-buf_500-sgd_l0_bfp1_fin_hw_noreplay-221026213641"),
            ("CIFAR100 SGD BFP L2", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar100-buf_500-sgd_l2_bfp1_fin_hw_noreplay-221027045156"),
        ]
    elif dataset_name == "cifar100_m2":
        ckpt_folders = [
            ("CIFAR100 SGD BFP L0", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar100-buf_500-sgd_l0_bfp1_fin_hw_noreplay-221026213641"),
        ]
    elif dataset_name == "cifar10_r1":
        ckpt_folders = [
            ("CIFAR10 SGD", "/home/qiao/src/mammoth/checkpoint/weights/sgd-seq-cifar10-221027205958",),
            ("CIFAR10 SGD BFP L1", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar10-buf_200-sgd_l1_bfp1_fin_hw_noreplay-221027012105"),
            ("CIFAR10 SGD BFP L0", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar10-buf_200-sgd_l0_bfp1_fin_hw_noreplay-221026213638"),
            ("CIFAR10 SGD BFP L2", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar10-buf_200-sgd_l2_bfp1_fin_hw_noreplay-221027045147"),
            ("CIFAR10 DER++", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar10-buf_200-derpp_bfp0_v2_ep50-220918165729",),
            ("CIFAR10 DER++ BFP L0", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar10-buf_200-derpp_l0_bfp1_hw_v2_ep50-221006160648",),
            ("CIFAR10 DER++ BFP L1", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar10-buf_200-derpp_l1_bfp1_hw_v2_ep50-221005213025",),
            ("CIFAR10 cumulative train", "/home/qiao/src/mammoth/checkpoint/weights/joint-seq-cifar10-221027224802"),
        ]
    elif dataset_name == "cifar100_r1":
        ckpt_folders = [
            ("CIFAR100 SGD", "/home/qiao/src/mammoth/checkpoint/weights/sgd-seq-cifar100-221020221356"),
            ("CIFAR100 SGD BFP L0", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar100-buf_500-sgd_l0_bfp1_fin_hw_noreplay-221026213641"),
            ("CIFAR100 SGD BFP L1", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar100-buf_500-sgd_l1_bfp1_fin_hw_noreplay-221027012112"),
            ("CIFAR100 SGD BFP L2", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar100-buf_500-sgd_l2_bfp1_fin_hw_noreplay-221027045156"),
            ("CIFAR100 DER++", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar100-buf_500-derpp_bfp0_v2_ep50-220919002820",),
            ("CIFAR100 DER++ BFP L0", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar100-buf_500-derpp_l0_bfp1_hw_v2_ep50-221006160704",),
            ("CIFAR100 DER++ BFP L1", "/home/qiao/src/mammoth/checkpoint/weights/bfp-seq-cifar100-buf_500-derpp_l1_bfp1_hw_v2_ep50-221005213029",),
            ("CIFAR100 cumulative train", "/home/qiao/src/mammoth/checkpoint/weights/sgd-seq-cifar100-cum_train-221021000620"),
        ]
    elif dataset_name == "tinyimg":
        ckpt_folders = [
            ["Tinyimg DER++", '/home/qiao/src/mammoth/checkpoint/supp-tinyimg/8789089/weights/bfp-seq-tinyimg-buf_4000-derpp_bfp0-221115212242'],
            ["Tinyimg DER++ BFP L0", '/home/qiao/src/mammoth/checkpoint/supp-tinyimg/8789090/weights/bfp-seq-tinyimg-buf_4000-derpp_l0_bfp1_fin_hw_v2-221115212236'],
            ["Tinyimg DER++ BFP L1", '/home/qiao/src/mammoth/checkpoint/supp-tinyimg/8789091/weights/bfp-seq-tinyimg-buf_4000-derpp_l1_bfp1_fin_hw_v2-221115212242'],
            ["Tinyimg DER++ BFP L2", '/home/qiao/src/mammoth/checkpoint/supp-tinyimg/8789092/weights/bfp-seq-tinyimg-buf_4000-derpp_l2_bfp1_fin_hw_v2-221115212242'],
            ["Tinyimg SGD", '/home/qiao/src/mammoth/checkpoint/supp-tinyimg/8787204/weights/bfp-seq-tinyimg-buf_4000-sgd-221115003840'],
            ["Tinyimg SGD BFP L0", '/home/qiao/src/mammoth/checkpoint/supp-tinyimg/8787205/weights/bfp-seq-tinyimg-buf_4000-sgd_l0_bfp1_fin_hw_noreplay-221115003840'],
            ["Tinyimg SGD BFP L1", '/home/qiao/src/mammoth/checkpoint/supp-tinyimg/8787206/weights/bfp-seq-tinyimg-buf_4000-sgd_l1_bfp1_fin_hw_noreplay-221115003840'],
            ["Tinyimg SGD BFP L2", '/home/qiao/src/mammoth/checkpoint/supp-tinyimg/8787207/weights/bfp-seq-tinyimg-buf_4000-sgd_l2_bfp1_fin_hw_noreplay-221115003840'],
        ]
    else:
        raise ValueError("Unknown dataset name: " + dataset_name)

    results = []

    for i in range(len(ckpt_folders)):
        name, ckpt_folder = ckpt_folders[i]
        dataset, models = get_model_dataset(ckpt_folder)
        train_dataset, test_dataset = dataset.get_datasets()
        results += study_linear_prob(models[-1].net, train_dataset, test_dataset, name, wandb_log=True)

    model_init = get_backbone(dataset.args, dataset)
    train_dataset, test_dataset = dataset.get_datasets()
    results += study_linear_prob(model_init, train_dataset, test_dataset, "random init", wandb_log=True)

    # Save linear probing results on cifar10 dataset to a pickle file
    with open("./notebooks/linear_probing_results_%s.pkl" % dataset_name, "wb") as f:
        pickle.dump(results, f)
