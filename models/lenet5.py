import torch
from torch import nn
from torch.nn import functional as F
import math


class LeNet5(nn.Module):

    def __init__(self, num_classes, image_channel, image_size):
        super(LeNet5, self).__init__()

        self.conv_unit = nn.Sequential(
            # x: [b, 1, 28, 28] -> [b, 6, 30, 30]
            nn.Conv2d(image_channel, 6, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),

            # x: [b, 6, 30, 30] -> [b, 6, 15, 15]
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            ##
            # x: [b, 6, 15, 15] -> [b, 16, 11, 11]
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        # flatten
        self.linear = nn.Linear(16 * math.floor((image_size - 6) / 4) ** 2, num_classes)

    def forward(self, x):
        x = self.conv_unit(x)
        x = x.view(x.size(0), -1)
        logits = self.linear(x)
        return logits


class LeNet5_backbone(nn.Module):

    def __init__(self, image_channel, image_size):
        super(LeNet5_backbone, self).__init__()

        self.conv_unit = nn.Sequential(
            # x: [b, 1, 28, 28] -> [b, 6, 30, 30]
            nn.Conv2d(image_channel, 6, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),

            # x: [b, 6, 30, 30] -> [b, 6, 15, 15]
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            ##
            # x: [b, 6, 15, 15] -> [b, 16, 11, 11]
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.feature_dim = 16 * math.floor((image_size - 6) / 4) ** 2

    def forward(self, x):
        x = self.conv_unit(x)
        feature = x.view(x.size(0), -1)
        return feature


def main():
    net = LeNet5(num_classes=10, image_channel=3, image_size=32)
    x = torch.randn((1, 3, 32, 32))
    out = net(x)
    print(out.shape)


if __name__ == '__main__':
    main()
