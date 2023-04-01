from ast import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import MammothBackbone, xavier, num_flat_features

class ConvNet(MammothBackbone):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()
        self.num_classes = num_classes
        self.net_width = net_width
        self.net_depth = net_depth
        self.net_act = net_act
        self.net_norm = net_norm
        self.net_pooling = net_pooling

        self._features, shape_feat = self._make_layers(
            channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
            
        feat_dim = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.feat_dim = feat_dim
        self.classifier = nn.Linear(feat_dim, num_classes)

        self._reset_parameters()

        # number of channels for each layer, used for bfp
        # Only consider the final layer
        self.net_channels = [self.feat_dim]
        
    
    def _reset_parameters(self):
        self.apply(xavier)

    def forward(self, x, returnt='out'):
        feats = self._features(x)
        feats = feats.contiguous().view(feats.size(0), -1)

        if returnt == 'features':
          return feats
        
        out = self.classifier(feats)

        if returnt == 'out':
          return out
        elif returnt == 'all':
          return (out, feats)

    def forward_all_layers(self, x: torch.Tensor):
        out, feats = self(x, returnt='all')
        return out, [feats]
          
    # def forward_all_layers(self, x):
    #     layer_depth = 2
    #     if self.net_norm != 'none':
    #         layer_depth +=1
    #     if self.net_pooling != 'none':
    #         layer_depth +=1
        
    #     feats = []
    #     out = x
    #     for i in range(self.net_depth):
    #         sub_net = self._features[i*layer_depth:(i+1)*layer_depth]
    #         out = sub_net(out)
    #         feats.append(out)
        
    #     out = out.view(out.size(0), -1)
    #     out = self.classifier(out)

    #     return out, feats

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2


        return nn.Sequential(*layers), shape_feat
