import torch
import torch.nn as nn
import torchvision.models as models
import os
import sys
import logging

sys.path.append('../')
from global_param import *

os.chdir(global_path)

os.environ['TORCH_HOME'] = global_path + '/pretrained'


class transformer_Swin_v2_t(nn.Module):
    def __init__(self, num_classes, pretrained: bool = False):
        super(transformer_Swin_v2_t, self).__init__()
        self.net = None
        if pretrained:
            weights = models.Swin_V2_T_Weights.DEFAULT
            self.net = models.swin_v2_t(weights=weights)
            # logger = logging.getLogger("mylogger")
            # logger.info("succeed to load the pretrained weights")
        else:
            self.net = models.swin_v2_t()
        self.net.head = nn.Linear(self.net.head.in_features, num_classes)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.net(x)
        return x


class transformer_Swin_v2_t_backbone(nn.Module):
    def __init__(self, pretrained: bool = False):
        super(transformer_Swin_v2_t_backbone, self).__init__()
        if pretrained:
            weights = models.Swin_V2_T_Weights.DEFAULT
            self.net = models.swin_v2_t(weights=weights)
            # logger = logging.getLogger("mylogger")
            # logger.info("succeed to load the pretrained weights")
        else:
            self.net = models.swin_v2_t()
        self.feature_dim = self.net.head.in_features
        self.net.head = torch.nn.Identity()

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        out = self.net(x)
        return out


def main():
    a = torch.randn((5, 3, 28, 28))
    net = transformer_Swin_v2_t(10, False)
    net2 = transformer_Swin_v2_t_backbone(False)
    b = net(a)
    c = net2(a)
    print(b.shape)
    print(c.shape)


if __name__ == '__main__':
    main()
