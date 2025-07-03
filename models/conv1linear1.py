import torch
from torch import nn
from torch.nn import functional as F
import math


class Conv1Linear1(nn.Module):

    def __init__(self, num_classes, image_channel, image_size):
        super(Conv1Linear1, self).__init__()

        self.conv_unit = nn.Sequential(
            # x: [b, 1, 28, 28] -> [b, 6, 30, 30]
            nn.Conv2d(image_channel, 6, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),

            # x: [b, 6, 30, 30] -> [b, 6, 15, 15]
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        )
        # flatten
        self.linear = nn.Linear(6 * math.floor((image_size + 2) / 2) ** 2, num_classes)

    def forward(self, x):
        x = self.conv_unit(x)
        x = x.view(x.size(0), -1)
        logits = self.linear(x)
        return logits


if __name__ == '__main__':
    for img_size in range(16, 128):
        model = Conv1Linear1(2, 1, img_size)
        a = torch.randn((4, 1, img_size, img_size))
        b = model(a)
        print(f"img_size: {img_size}, {b.shape}")
