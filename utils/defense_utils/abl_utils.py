import torch
import logging

logger = logging.getLogger("mylogger")


def scheduler_pretrain(optimizer, epoch, args):
    """
    set learning rate during the process of pre-train model
    """
    if epoch < args.pretrain_epochs:
        lr = args.lr_pretrain
    else:
        lr = 0.01
    logger.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def scheduler_unlearning(optimizer, epoch, args):
    """
    set learning rate during the process of unlearning model
    """
    if epoch < args.unlearning_epochs:
        lr = args.lr_unlearning
    else:
        lr = 0.0001
    logger.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def scheduler_finetuning(optimizer, epoch, args):
    """
    set learning rate during the process of finetuning model
    """
    if epoch < 40:
        lr = 0.01
    elif epoch < 60:
        lr = 0.001
    else:
        lr = 0.001
    logger.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_classification_acc(poisoned_dataset=None, clean_dataset=None):
    """
    Compute the TP and FP of poisoned samples classification
    :param poisoned_dataset: dataset_wrapper_with_transform, (img, label, original_index, poison_or_not, original_target)
    :return: TP, FP
    """
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    if poisoned_dataset is not None:
        for sample in poisoned_dataset:
            if sample[3] == 1:
                TP += 1
            elif sample[3] == 0:
                FP += 1
            else:
                raise ValueError("Error in sample")

    if clean_dataset is not None:
        for sample in clean_dataset:
            if sample[3] == 1:
                FN += 1
            elif sample[3] == 0:
                TN += 1
            else:
                raise ValueError("Error in sample")
    return TP, FP, FN, TN

