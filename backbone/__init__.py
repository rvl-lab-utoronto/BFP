# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple
import torch
import torch.nn as nn


def get_classifier(type, num_feat: int, num_classes) -> nn.Module:
    if type == "linear":
        classifier = nn.Linear(num_feat, num_classes)
    elif type == "mlp-2":
        classifier = nn.Sequential(
            nn.Linear(num_feat, max(num_classes, num_feat // 4)),
            nn.ReLU(),
            nn.Linear(max(num_classes, num_feat // 4), num_classes)
        )
    else:
        raise NotImplementedError("Unknown classifier type: %s"%type)
    
    return classifier

def xavier(m: nn.Module) -> None:
    """
    Applies Xavier initialization to linear modules.

    :param m: the module to be initialized

    Example::
        >>> net = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        >>> net.apply(xavier)
    """
    if m.__class__.__name__ == 'Linear':
        fan_in = m.weight.data.size(1)
        fan_out = m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def num_flat_features(x: torch.Tensor) -> int:
    """
    Computes the total number of items except the first dimension.

    :param x: input tensor
    :return: number of item from the second dimension onward
    """
    size = x.size()[1:]
    num_features = 1
    for ff in size:
        num_features *= ff
    return num_features

class MammothBackbone(nn.Module):

    def __init__(self, **kwargs) -> None:
        super(MammothBackbone, self).__init__()

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        raise NotImplementedError

    def forward_all_layers(self, x: torch.Tensor) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        raise NotImplementedError

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, returnt='features')

    def freeze_feature(self) -> None:
        print("Backbone: Freezing feature")
        if isinstance(self._features, list):
            for ff in self._features:
                for pp in list(ff.parameters()):
                    pp.requires_grad = False
        else:
            for pp in list(self._features.parameters()):
                pp.requires_grad = False

    def unfreeze_feature(self) -> None:
        print("Backbone: Unfreezing feature")
        if isinstance(self._features, list):
            for ff in self._features:
                for pp in list(ff.parameters()):
                    pp.requires_grad = True
        else:
            for pp in list(self._features.parameters()):
                pp.requires_grad = True

    def freeze_classifier(self) -> None:
        print("Backbone: Freezing classifier")
        for pp in list(self.classifier.parameters()):
            pp.requires_grad = False

    def unfreeze_classifier(self) -> None:
        print("Backbone: Unfreezing classifier")
        for pp in list(self.classifier.parameters()):
            pp.requires_grad = True

    def unfreeze(self) -> None:
        self.unfreeze_feature()
        self.unfreeze_classifier()
    
    def reinit_classifier(self) -> None:
        self.classifier.apply(xavier)
    
    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        return torch.cat(self.get_grads_list())
    
    def get_grads_list(self):
        """
        Returns a list containing the gradients (a tensor for each layer).
        :return: gradients list
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return grads
