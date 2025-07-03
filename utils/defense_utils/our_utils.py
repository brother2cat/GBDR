import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F


def compute_cross_poisoned_num(bd_dataset1, bd_dataset2):
    """
    Compute the num of poisoned samples that both bd_dataset1 and bd_dataset2 classify as poisoned samples
    :param bd_dataset1:
    :param bd_dataset2:
    :return:
    """
    original_index_1 = []
    original_index_2 = []
    for sample in bd_dataset1:
        if sample[3] == 1:
            original_index_1.append(sample[2])

    for sample in bd_dataset2:
        if sample[3] == 1:
            original_index_2.append(sample[2])

    intersection = list(set(original_index_1) & set(original_index_2))
    return len(intersection)


def dict_value_normalized(dict_example):
    """
    normalized the value of dict
    :param dict_example:
    :return:
    """
    values = list(dict_example.values())
    mean = np.mean(values)
    std = np.std(values)
    for key in dict_example:
        dict_example[key] = (dict_example[key] - mean) / std
    return dict_example


class Mix_adversarial_Dataset(Dataset):
    def __init__(self, dataset1, dataset2):
        """

        :param dataset1: loss is set to normal
        :param dataset2: loss is set to special
        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1 = len(self.dataset1)
        self.len2 = len(self.dataset2)

    def __getitem__(self, index):
        if index < self.len1:
            return self.dataset1[index] + (0,)
        else:
            return self.dataset2[index - self.len1] + (1,)

    def __len__(self):
        return self.len1 + self.len2


class ADV_loss(nn.Module):
    # use the Mix +-CrossEntropy loss to backward
    def __init__(self):
        super(ADV_loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        return

    def forward(self, output, mask, target):
        loss = self.criterion(output, target)
        loss_special = -torch.dot(loss, mask)
        mask_normal = 1 - mask
        loss_normal = torch.dot(loss, mask_normal)
        loss_avg = (loss_special + loss_normal) / len(target)
        return loss_avg


class Adv_finetune_loss(nn.Module):
    # use the -CrossEntropy loss to backward
    def __init__(self):
        super(Adv_finetune_loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        return

    def forward(self, output, target):
        loss = self.criterion(output, target)
        # add Local Gradient Ascent(LGA) loss
        loss_ascent = -loss
        return loss_ascent


def scheduler_adv_finetune(optimizer, epoch, args, logger, num_iter):
    """
    set learning rate during the process of adv finetune
    """
    if epoch < args.adv_finetune_epochs:
        lr = args.lr_adv_finetune
    else:
        lr = 0.0001
    step_total = len(args.steplr_finetune_milestones)
    for i in range(step_total):
        if num_iter >= args.steplr_finetune_milestones[i]:
            lr *= args.steplr_finetune_gamma
        else:
            break
    logger.info('Adv_Finetune------epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def scheduler_clean_purity_finetune(optimizer, epoch, args, logger, num_iter):
    """
    set learning rate during the process of clean purity finetune
    """
    if epoch < args.clean_purity_finetune_epochs:
        lr = args.lr_clean_purity_finetune
    else:
        lr = 0.0001
    step_total = len(args.steplr_finetune_milestones)
    for i in range(step_total):
        if num_iter >= args.steplr_finetune_milestones[i]:
            lr *= args.steplr_finetune_gamma
        else:
            break
    logger.info('Clean_purity_Finetune------epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Mix_two_Dataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1 = len(self.dataset1)
        self.len2 = len(self.dataset2)

    def __getitem__(self, index):
        if index < self.len1:
            return self.dataset1[index]
        else:
            return self.dataset2[index - self.len1]

    def __len__(self):
        return self.len1 + self.len2


def pad28to32with0(images: torch.tensor):
    pad = (2, 2, 2, 2)
    images = F.pad(images, pad, "constant", value=0.)
    return images


def crop32to28(images: torch.tensor):
    images = images[:, :, 2:30, 2:30]
    return images
