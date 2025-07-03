import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F

import sys
import os

sys.path.append('../')
from global_param import *

os.chdir(global_path)


class CHW_np_2_Tensor:
    def __call__(self, x):
        return torch.from_numpy(x)


class general_np2tensor:
    def __init__(self):
        self.np_hwc_2_tensor = transforms.ToTensor()
        self.np_chw_2_tensor = CHW_np_2_Tensor()

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            if x.shape[0] <= 3:
                return self.np_chw_2_tensor(x)
            else:
                return self.np_hwc_2_tensor(x)
        else:
            return self.np_hwc_2_tensor(x)


class transforms_image:
    mnist_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    cifar_train = transforms.Compose([
        general_np2tensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    cifar_test = transforms.Compose([
        general_np2tensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    cifar100_train = transforms.Compose([
        general_np2tensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    cifar100_test = transforms.Compose([
        general_np2tensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    imagenet_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    imagenet_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def get_dataset_and_transform(dataset_name):
    """

    :param dataset_name: mnist ,cifar10, mini
    :return: train_dataset_wo_transform, train_transform, test_dataset_wo_transform, test_transform
    """
    if dataset_name == 'mnist':
        mnist_train = datasets.MNIST('datasets/mnist', train=True, transform=None, download=True)
        mnist_test = datasets.MNIST('datasets/mnist', train=False, transform=None, download=True)
        return mnist_train, transforms_image.mnist_train, mnist_test, transforms_image.mnist_test

    elif dataset_name == 'cifar10':
        cifar_train = datasets.CIFAR10('datasets/cifar10', train=True, transform=None, download=True)
        cifar_test = datasets.CIFAR10('datasets/cifar10', train=False, transform=None, download=True)
        return cifar_train, transforms_image.cifar_train, cifar_test, transforms_image.cifar_test

    elif dataset_name == 'cifar100':
        cifar_train = datasets.CIFAR100('datasets/cifar100', train=True, transform=None, download=True)
        cifar_test = datasets.CIFAR100('datasets/cifar100', train=False, transform=None, download=True)
        return cifar_train, transforms_image.cifar100_train, cifar_test, transforms_image.cifar100_test

    elif dataset_name == 'gtsrb':
        gtsrb_train = datasets.GTSRB('datasets/gtsrb', split="train", transform=None, download=True)
        gtsrb_test = datasets.GTSRB('datasets/gtsrb', split="test", transform=None, download=True)
        return gtsrb_train, transforms_image.cifar_train, gtsrb_test, transforms_image.cifar_test

    elif dataset_name == 'mini':
        train_set = datasets.ImageFolder("datasets/mini/train", transform=None)
        test_set = datasets.ImageFolder("datasets/mini/val", transform=None)
        return train_set, transforms_image.imagenet_train, test_set, transforms_image.imagenet_test
    else:
        raise ValueError("mnist ,cifar10, mini")


def main():
    mnist_train, _, _, _ = get_dataset_and_transform("mnist")
    a = mnist_train[0][0]
    print(a)


if __name__ == '__main__':
    main()
