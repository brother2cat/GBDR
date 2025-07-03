import torch
from torch import tensor


def de_normalize(dataset, norm_image: tensor):
    if dataset == "mnist":
        mean = 0.1307
        std = 0.3081
    elif dataset == "cifar10":
        mean = torch.tensor((0.4914, 0.4822, 0.4465))
        std = torch.tensor((0.2470, 0.2435, 0.2616))
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    elif dataset == "cifar100":
        mean = torch.tensor((0.5071, 0.4865, 0.4409))
        std = torch.tensor((0.2673, 0.2564, 0.2762))
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    elif dataset == "mini":
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:
        raise ValueError(f"dataset {dataset} must in [mnist, cifar10, mini]")

    return norm_image * std + mean


if __name__ == '__main__':
    a = torch.ones((2, 3, 1, 1))
    b = de_normalize('cifar10', a)
    print(b)

