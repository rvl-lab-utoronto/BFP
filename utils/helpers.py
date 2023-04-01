import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import backbone

from datasets import get_dataset
from models import get_model

from utils.routines import forward_loader_all_layers
from utils.main import get_backbone

def project_onto(X, V):
    '''
    Input X: (N, D)
    Input V: (D, K)
    '''
    X_mean = X.mean(0)
    X = X - X_mean
    X_proj = X @ V
    X_proj = X_proj + X_mean
    return X_proj

def evaluate_clf(feat, y, classifier):
    classifier.eval()
    classifier.to(feat.device)
    with torch.no_grad():
        logits = classifier(feat)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
    return loss, acc

def load_models(ckpt_folder, n_models=1, update_args=None):
    models = []
    for i in range(n_models):
        model, dataset, args = load_checkpoint(ckpt_folder, i, update_args)
        models.append(model)

    return models, dataset, args

def compute_feat_svd(model, loader):
    logits, feats, xs, ys = forward_loader_all_layers(model.net, loader)
    feats = feats[-1].mean((2,3))
    U, S, V = torch.svd(feats - feats.mean(0, keepdim=True), compute_uv=True)
    return U, S, V

def compute_svd(x):
    x_mean = x.mean(0, keepdim=True)
    U, S, V = torch.svd(x - x_mean, compute_uv=True)
    return U, S, V, x_mean

def load_checkpoint(ckpt_folder, task_id, update_args=None):
    ckpt_path = os.path.join(ckpt_folder, "model_%d.ckpt" % task_id)
    # print("Loading model weights from", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt['args']

    if update_args is not None:
        update_args(args)

    dataset = get_dataset(args)
    loss = dataset.get_loss()

    # backbone = dataset.get_backbone()
    backbone = get_backbone(args, dataset)
    model = get_model(args, backbone, loss, dataset.get_transform())
    state_dict = ckpt['state_dict']
    buffer0 = clean_state_dict(state_dict)
    model.load_state_dict(ckpt['state_dict'], strict=False)

    return model, dataset, args

def clean_state_dict(state_dict):
    '''
    Remove the old_net and extract the buffer from the checkpoint
    '''
    buffer = {}
    for k in list(state_dict.keys()):
        if k.startswith("old_net."):
            del state_dict[k]
        if k.startswith("buffer."):
            buffer[k[7:]] = state_dict.pop(k)
            
    return buffer