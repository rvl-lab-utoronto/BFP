# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List
from backbone import MammothBackbone

def getNormLayer(norm, n_feats, shape=None):
    if norm == 'instancenorm':
        layer = nn.GroupNorm(n_feats, n_feats, affine=True)
    elif norm == 'groupnorm':
        layer = nn.GroupNorm(4, n_feats, affine=True)
    elif norm == 'batchnorm':
        layer = nn.BatchNorm2d(n_feats)
    elif norm == "layernorm":
        # TODO: track the feature shape in ResNet18 and pass it here
        assert shape is not None
        layer = nn.LayerNorm(n_feats)
    elif norm is None:
        layer = nn.Identity()
    else:
        raise NotImplementedError("Unknown norm layer: %s"%norm)

    return layer

def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1, 
                 skip_relu:bool = False, norm:str = 'batchnorm') -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = getNormLayer(norm, planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = getNormLayer(norm, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                getNormLayer(norm, self.expansion * planes)
            )

        self.skip_relu = skip_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        if not self.skip_relu:
            out = relu(out)

        return out


class ResNet(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, norm='batchnorm') -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.norm = norm

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = getNormLayer(norm, nf * 1)

        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

        self.feat_dim = nf * 8 * block.expansion

        self._features = [self.conv1,
            self.bn1,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        ]
        self.classifier = self.linear
        self.feat2logits = None

        # number of channels for each layer, used for bfp
        self.net_channels = [nf * 1, nf * 2, nf * 4, nf * 8]

    def skip_relu(self, skip=True, last=False):
        if last:
            self.layer4[-1].skip_relu = skip
        else:
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                layer[-1].skip_relu = skip
                

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm=self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def add_gm_loss(self, gm_loss):
        self.feat2logits = lambda x: gm_loss(x)[0]


    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :param returnt: return type (a string among 'out', 'features', 'all')
        :return: output tensor (output_classes)
        """
        
        out = relu(self.bn1(self.conv1(x))) # 64, 32, 32
        if hasattr(self, 'maxpool'):
            out = self.maxpool(out)
        out = self.layer1(out)  # -> 64, 32, 32
        out = self.layer2(out)  # -> 128, 16, 16
        out = self.layer3(out)  # -> 256, 8, 8
        out = self.layer4(out)  # -> 512, 4, 4
        out = avg_pool2d(out, out.shape[2]) # -> 512, 1, 1
        feature = out.view(out.size(0), -1)  # 512

        if returnt == 'features':
            return feature

        if self.feat2logits is None:
            out = self.classifier(feature)
        else:
            out = self.feat2logits(feature)
        
        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, feature)
        
        raise NotImplementedError("Unknown return type")

    def forward_all_layers(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        
        out = relu(self.bn1(self.conv1(x))) # 64, 32, 32
        if hasattr(self, 'maxpool'):
            out = self.maxpool(out)
        out = self.layer1(out)  # -> 64, 32, 32
        feats.append(out)

        out = self.layer2(out)  # -> 128, 16, 16
        feats.append(out)

        out = self.layer3(out)  # -> 256, 8, 8
        feats.append(out)

        out = self.layer4(out)  # -> 512, 4, 4
        feats.append(out)

        out = avg_pool2d(out, out.shape[2]) # -> 512, 1, 1
        feature = out.view(out.size(0), -1)  # 512
        
        out = self.classifier(feature)

        return out, feats


def resnet18(nclasses: int, args) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    nf = 64
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, norm=args.backbone_norm)
