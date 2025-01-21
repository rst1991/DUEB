# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../furnace/")

from functools import partial
from collections import OrderedDict
from config import config
from base_model import resnet50
#from base_model import resnet101

class bayesian_categorical_crossentropy(nn.Module):
    def __init__(self, T, num_classes):
        super(bayesian_categorical_crossentropy, self).__init__()
        self.T = T
        self.ELU = nn.ELU()
        self.num_classes = num_classes
        self.categorical_crossentropy = nn.CrossEntropyLoss(ignore_index=255)

    def bayesian_categorical_crossentropy_internal(self, logit, var, true):

        std = torch.sqrt(var) + 1e-15
        variance_depressor = torch.exp(var) - torch.ones_like(var)
        undistorted_loss = self.categorical_crossentropy(logit+1e-15,true) #In pytorch loss (output,target)
        
        #iterable = torch.autograd.Variable(np.ones(self.T))
        #print(std.shape, std.dtype)
        dist = torch.distributions.normal.Normal(torch.zeros_like(std), std)
        monte_carlo = [self.gaussian_categorical_crossentropy(logit, true, dist, undistorted_loss, self.num_classes) for _ in range(self.T)]
        monte_carlo = torch.stack(monte_carlo)
        variance_loss = torch.mean(monte_carlo,axis = 0) * undistorted_loss
        
        loss_final = variance_loss + undistorted_loss + variance_depressor
        # reduction of loss required. Taking mean() as that is what happens in batched crossentropy
        return loss_final.mean()
    
    def gaussian_categorical_crossentropy(self, logit, true, dist, undistorted_loss, num_classes):
        std_samples = torch.squeeze(torch.transpose(dist.sample((num_classes,)), 0,1))
        #print("########### pred",pred.shape," std samples ",std_samples.shape)
        distorted_loss = self.categorical_crossentropy(logit + 1e-15 + std_samples, true)
        diff = undistorted_loss - distorted_loss
        return -1*self.ELU(diff)
    
    def forward(self, logit, var, true):
        return self.bayesian_categorical_crossentropy_internal(logit, var, true)

class Network(nn.Module):
    def __init__(self, num_classes, criterion, norm_layer, pretrained_model=None):
        super(Network, self).__init__()
        self.branch1 = SingleNetwork(num_classes, criterion, norm_layer, pretrained_model)
        self.branch2 = SingleNetwork(num_classes, criterion, norm_layer, pretrained_model)

    def forward(self, data, step=1):
        if not self.training:
            pred1 = self.branch1(data)
            pred2 = self.branch2(data)
            pred3 = (pred1 + pred2)/2
            return pred1

        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)

class SingleNetwork(nn.Module):
    def __init__(self, num_classes, criterion, norm_layer, pretrained_model=None):
        super(SingleNetwork, self).__init__()
        self.backbone = resnet50(pretrained_model, norm_layer=norm_layer,    #change resnet
                                  bn_eps=config.bn_eps,
                                  bn_momentum=config.bn_momentum,
                                  deep_stem=True, stem_width=64)
        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

        self.head = Head(num_classes, norm_layer, config.bn_momentum)
        self.business_layer = []
        self.business_layer.append(self.head)
        self.criterion = criterion

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        self.business_layer.append(self.classifier)

        self.variancelayer=nn.Conv2d(256, 1, kernel_size=1, bias=True)
        self.softplus=nn.Softplus()

    def forward(self, data):
        blocks = self.backbone(data)
        v3plus_feature = self.head(blocks)      # (b, c, h, w)
        b, c, h, w = v3plus_feature.shape

        pred = self.classifier(v3plus_feature)
        var=self.variancelayer(v3plus_feature)
        varp=self.softplus(var)

        b, c, h, w = data.shape
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
        varp = F.interpolate(varp, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            return v3plus_feature, pred, varp
        return pred

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)       # add activation layer
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)

        pool = self.leak_relu(pool)  # add activation layer

        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)  # add activation layer
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (min(try_index(self.pooling_size, 0), x.shape[2]),
                            min(try_index(self.pooling_size, 1), x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            )

            pool = nn.functional.avg_pool2d(x, pooling_size, stride=1)
            pool = nn.functional.pad(pool, pad=padding, mode="replicate")
        return pool

class Head(nn.Module):
    def __init__(self, classify_classes, norm_act=nn.BatchNorm2d, bn_momentum=0.0003):
        super(Head, self).__init__()

        self.classify_classes = classify_classes
        self.aspp = ASPP(2048, 256, [6, 12, 18], norm_act=norm_act)

        self.reduce = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            norm_act(48, momentum=bn_momentum),
            nn.ReLU(),
        )
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       )

    def forward(self, f_list):
        f = f_list[-1]
        f = self.aspp(f)

        low_level_features = f_list[0]
        low_h, low_w = low_level_features.size(2), low_level_features.size(3)
        low_level_features = self.reduce(low_level_features)

        f = F.interpolate(f, size=(low_h, low_w), mode='bilinear', align_corners=True)
        f = torch.cat((f, low_level_features), dim=1)
        f = self.last_conv(f)

        return f


if __name__ == '__main__':
    model = Network(40, criterion=nn.CrossEntropyLoss(),
                    pretrained_model=None,
                    norm_layer=nn.BatchNorm2d)
    left = torch.randn(2, 3, 128, 128)
    right = torch.randn(2, 3, 128, 128)

    print(model.backbone)

    out = model(left)
    print(out.shape)
