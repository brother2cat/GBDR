import torch
import torch.optim.lr_scheduler as lrs
import logging


def args_check(args):
    if args.dataset not in ['mnist', 'cifar10', 'mini', 'cifar100']:
        raise ValueError(f"{args.dataset} is not one of ['mnist', 'cifar10', 'mini', 'cifar100']")
    if args.model not in ['lenet5', 'resnet18', 'resnet34', 'resnet50', 'WRN28', 'swin_v2_t', 'conv1linear1']:
        raise ValueError(f"{args.model} is not one of ['lenet5', 'resnet18', 'resnet34', "
                         f"'resnet50', 'WRN28', 'conv1linear1', 'swin_v2_t']")
    if args.model == 'WRN28' and args.drop_rate is None:
        raise ValueError("When model is WRN28, drop_rate should not be None")
    if args.model == 'swin_v2_t' and args.pretrained is None:
        raise ValueError("When model is swin_v2_t, pretrained should not be None")
    if args.batch_size <= 0:
        raise ValueError("batch_size should > 0")
    if args.device != 'cpu' and not args.device.startswith("cuda:"):
        raise ValueError("device should start with 'cuda:' or be 'cpu'")
    if args.optimizer not in ['sgd', 'adamw']:
        raise ValueError("optimizer should in ['sgd', 'adamw']")
    if args.lr_scheduler not in ['steplr', 'multisteplr', 'cosine'] and args.lr_scheduler is not None:
        raise ValueError("lr_scheduler should be None or in ['steplr', 'multisteplr', 'cosine']")
    if args.lr_scheduler == 'steplr' and (args.steplr_stepsize is None or args.steplr_gamma is None):
        raise ValueError("When lr_scheduler is steplr, stepsize and steplr_gamma should not be None")
    if args.lr_scheduler == 'multisteplr' and (args.steplr_milestones is None or args.steplr_gamma is None):
        raise ValueError("When lr_scheduler is multisteplr, milestones and steplr_gamma should not be None")
    if args.lr_scheduler == 'cosine' and (args.steplr_T_max is None or args.steplr_eta_min is None):
        raise ValueError("When lr_scheduler is cosine, T_max and steplr_eta_min should not be None")
    if args.save_folder is None:
        raise ValueError("save folder can not be None")


class None_class:
    def step(self):
        pass


def get_lr_scheduler(args, optimizer, start_epoch=-1):
    if args.lr_scheduler is None:
        return None_class()
    elif args.lr_scheduler == "step_lr":
        return lrs.StepLR(optimizer, args.steplr_stepsize, args.steplr_gamma, last_epoch=start_epoch)
    elif args.lr_scheduler == "multisteplr":
        return lrs.MultiStepLR(
            optimizer, milestones=args.steplr_milestones, gamma=args.steplr_gamma, last_epoch=start_epoch)
    elif args.lr_scheduler == "cosine":
        return lrs.CosineAnnealingLR(
            optimizer, T_max=args.steplr_T_max, eta_min=args.steplr_eta_min, last_epoch=start_epoch)
    else:
        raise ValueError("lr_scheduler should be None or in ['steplr', 'multisteplr', 'cosine']")


def get_optimizer(model, args):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate,
                                    momentum=args.sgd_momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.learning_rate,
                                      betas=(0.9, 0.999),
                                      weight_decay=args.weight_decay)
    return optimizer
