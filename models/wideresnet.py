import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class WideResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, drop_rate=0.0):
        super(WideResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return out


class StageBlock(nn.Module):
    def __init__(self, WideResBlock_num, in_channels, out_channels, stride, drop_rate=0.0):
        super(StageBlock, self).__init__()
        self.stage_layer = self._make_layer(WideResBlock_num, in_channels, out_channels, stride, drop_rate)

    @staticmethod
    def _make_layer(WideResBlock_num, in_channels, out_channels, stride, drop_rate):
        layers = []
        for i in range(int(WideResBlock_num)):
            if i == 0:
                layers.append(WideResBlock(in_channels, out_channels, stride, drop_rate))
            else:
                layers.append(WideResBlock(out_channels, out_channels, 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.stage_layer(x)


class WideResNet(nn.Module):
    def __init__(self, dataset, depth, num_classes, widen_factor=1, drop_rate=0.5):
        super(WideResNet, self).__init__()
        stage_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6

        if dataset == 'mnist':
            self.conv1 = nn.Conv2d(1, stage_channels[0], kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, stage_channels[0], kernel_size=3, stride=1, padding=1)
        # 1st block
        self.block1 = StageBlock(n, stage_channels[0], stage_channels[1], 1, drop_rate)
        # 2nd block
        self.block2 = StageBlock(n, stage_channels[1], stage_channels[2], 2, drop_rate)
        # 3rd block
        self.block3 = StageBlock(n, stage_channels[2], stage_channels[3], 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(stage_channels[3], momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.global_avgpool = nn.AvgPool2d(8)
        if dataset == "cifar10" or dataset == "cifar100":
            self.fc_input = stage_channels[3] * 1
        elif dataset == "mini":
            self.fc_input = stage_channels[3] * 2 * 2
        elif dataset == 'mnist':
            self.fc_input = stage_channels[3] * 1
        else:
            assert False, "If you want to use wideresnet, the dataset must be cifar10 or mini"
        self.fc = nn.Linear(self.fc_input, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn1(out)
        out = self.relu(out)
        if out.shape[2] == 7:
            out = F.pad(out, (0, 1, 0, 1), "constant", 0)
        out = self.global_avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class WideResNet_backbone(nn.Module):
    def __init__(self, dataset, depth, widen_factor=1, drop_rate=0.5):
        super(WideResNet_backbone, self).__init__()
        stage_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6

        self.conv1 = nn.Conv2d(3, stage_channels[0], kernel_size=3, stride=1, padding=1)
        # 1st block
        self.block1 = StageBlock(n, stage_channels[0], stage_channels[1], 1, drop_rate)
        # 2nd block
        self.block2 = StageBlock(n, stage_channels[1], stage_channels[2], 2, drop_rate)
        # 3rd block
        self.block3 = StageBlock(n, stage_channels[2], stage_channels[3], 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(stage_channels[3], momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.global_avgpool = nn.AvgPool2d(8)
        if dataset == "cifar10" or dataset == "cifar100":
            self.feature_dim = stage_channels[3] * 1
        elif dataset == "mini":
            self.feature_dim = stage_channels[3] * 2 * 2
        else:
            assert False, "If you want to use wideresnet, the dataset must be cifar10 or mini"

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.global_avgpool(out)
        out = out.view(out.size(0), -1)
        return out


if __name__ == '__main__':
    net1 = WideResNet("mnist", 28, num_classes=10, widen_factor=10, drop_rate=0.3)
    a = torch.randn((2, 3, 28, 28))
    b = net1(a)

    print(b.shape)

