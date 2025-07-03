import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from argparse import Namespace

sys.path.append('../')
from global_param import *

os.chdir(global_path)

from models.lenet5 import LeNet5, LeNet5_backbone
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet18_backbone, ResNet34_backbone, ResNet50_backbone
from models.wideresnet import WideResNet, WideResNet_backbone
from models.swin import transformer_Swin_v2_t, transformer_Swin_v2_t_backbone
from models.conv1linear1 import Conv1Linear1


def get_model(model_name, args):
    if model_name == 'lenet5':
        model = LeNet5(num_classes=args.num_classes, image_channel=args.channel, image_size=args.image_size)
    elif model_name == 'resnet18':
        model = ResNet18(num_classes=args.num_classes, image_channel=args.channel, image_size=args.image_size)
    elif model_name == 'resnet34':
        model = ResNet34(num_classes=args.num_classes, image_channel=args.channel, image_size=args.image_size)
    elif model_name == 'resnet50':
        model = ResNet50(num_classes=args.num_classes, image_channel=args.channel, image_size=args.image_size)
    elif model_name == 'WRN28':
        model = WideResNet(args.dataset, 28, num_classes=args.num_classes, widen_factor=10, drop_rate=args.drop_rate)
    elif model_name == 'swin_v2_t':
        model = transformer_Swin_v2_t(num_classes=args.num_classes, pretrained=args.pretrained)
    elif model_name == 'conv1linear1':
        model = Conv1Linear1(num_classes=args.num_classes, image_channel=args.channel, image_size=args.image_size)
    else:
        raise ValueError(f"We do not have the model-{model_name}")
    return model


def get_model_backbone(args):
    if args.model == 'lenet5':
        model = LeNet5_backbone(image_channel=args.channel, image_size=args.image_size)
    elif args.model == 'resnet18':
        model = ResNet18_backbone(image_channel=args.channel, image_size=args.image_size)
    elif args.model == 'resnet34':
        model = ResNet34_backbone(image_channel=args.channel, image_size=args.image_size)
    elif args.model == 'resnet50':
        model = ResNet50_backbone(image_channel=args.channel, image_size=args.image_size)
    elif args.model == 'WRN28':
        model = WideResNet_backbone(args.dataset, 28, widen_factor=10, drop_rate=args.drop_rate)
    else:
        model = transformer_Swin_v2_t_backbone(pretrained=args.pretrained)
    return model


class SelfModel(nn.Module):
    def __init__(self, args: Namespace, backbone):
        super(SelfModel, self).__init__()
        self.backbone = backbone

        if args.head == "linear":
            self.proj_head = nn.Linear(self.backbone.feature_dim, args.projected_dim)
        elif args.head == "mlp":
            self.proj_head = nn.Sequential(
                nn.Linear(self.backbone.feature_dim, self.backbone.feature_dim),
                nn.BatchNorm1d(self.backbone.feature_dim),
                nn.ReLU(),
                nn.Linear(self.backbone.feature_dim, args.projected_dim),
            )
        else:
            raise ValueError("Invalid head {}".format(args.head))

    def forward(self, x):
        feature = self.proj_head(self.backbone(x))
        feature = F.normalize(feature, dim=1)

        return feature


class LinearModel(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


class Full_Model(nn.Module):
    def __init__(self, backbone, linear):
        super(Full_Model, self).__init__()
        self.backbone = backbone
        self.linear = linear

    def forward(self, x):
        out = self.backbone(x)
        out = self.linear(out)
        return out


if __name__ == '__main__':
    class args:
        def __init__(self):
            self.projected_dim = 128
            self.head = "mlp"
            self.model = "resnet18"
            self.dataset = "cifar10"
            self.image_size = 32
            self.channel = 3



