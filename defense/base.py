import torch
import logging
import yaml
import sys
import os
from copy import deepcopy
import argparse

sys.path.append('../')
from global_param import *

os.chdir(global_path)

from datasets.get_dataset import get_dataset_and_transform
from datasets.ImageDataInfo import *
from models.get_model import get_model
from utils.path_check import path_check
from utils.fix_random import fix_random
from utils.args_check import args_check, get_lr_scheduler
from utils.dataset_wrapper import dataset_wrapper_with_transform
from utils.backdoor_dataset import prepare_bd_Dataset


class defense:
    def __init__(self, attack_yaml_path):
        # load attack parameters
        parser = argparse.ArgumentParser(description=sys.argv[0])
        parser.add_argument("-d", "--dataset", type=str, help="which dataset to use")
        parser.add_argument("-ya", "--attack_yaml_path", type=str, help="attack config file")
        parser.add_argument("-yd", "--defense_yaml_path", type=str, help="defense config file")
        parser.add_argument("-p", "--defense_save_folder_path", type=str, help="defense")
        parser.add_argument("-dv", '--device', type=str)
        self.args = parser.parse_args()
        if self.args.attack_yaml_path is None:
            pass
        else:
            attack_yaml_path = self.args.attack_yaml_path
        with open(attack_yaml_path, 'r') as f:
            bd_defaults = yaml.safe_load(f)
        self.args.__dict__.update({k: v for k, v in bd_defaults.items() if v is not None and
                                   (k not in self.args.__dict__ or self.args.__dict__[k] is None)})

        # set dataset info
        if self.args.dataset == "mnist":
            temp_class = mnistInfo
        elif self.args.dataset == "cifar10":
            temp_class = cifar10Info
        elif self.args.dataset == "cifar100":
            temp_class = cifar100Info
        else:
            temp_class = miniInfo
        self.args.num_classes = temp_class.num_classes
        self.args.channel = temp_class.channel
        self.args.image_size = temp_class.height

        # self.save_folder is the save folder of defense
        # self.args.save_folder is the save folder of backdoor attack
        # need to reload this parameters
        self.save_folder = None

        self.logger = None
        self.clean_test_image_transform = None
        self.clean_test_dataset_wo_transform = None
        self.clean_train_image_transform = None
        self.clean_train_dataset_wo_transform = None

        self.bd_train_dataset_wo_transform = None
        self.bd_test_dataset_wo_transform = None

        self.bd_model = None
        self.device = None

    def backdoor_prepare(self):
        # set logger
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        self.logger = logging.getLogger("mylogger")
        self.logger.setLevel(logging.DEBUG)
        # file Handler
        path_check(self.save_folder)
        fileHandler = logging.FileHandler(self.save_folder + '/debug.log')
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(logging.DEBUG)
        self.logger.addHandler(fileHandler)
        # consoleHandler
        consoleHandler = logging.StreamHandler(sys.stderr)
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(logging.INFO)
        self.logger.addHandler(consoleHandler)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        self.logger.debug("Only INFO or above level log will show in cmd. DEBUG level log only will show in log file.")

        # set the random seed
        if self.args.random_seed is not None:
            fix_random(int(self.args.random_seed))
            self.logger.debug("fix randomness")

        # load clean dataset and transform
        self.clean_train_dataset_wo_transform, \
            self.clean_train_image_transform, \
            self.clean_test_dataset_wo_transform, \
            self.clean_test_image_transform = get_dataset_and_transform(self.args.dataset)
        self.logger.debug(f"succeed to load {self.args.dataset} clean dataset and transform")

        # load backdoor dataset
        self.bd_train_dataset_wo_transform = prepare_bd_Dataset(
            deepcopy(self.clean_train_dataset_wo_transform), train=True)
        state_dict = torch.load(global_path + "/" + self.args.save_folder + "/bd_dataset/train_bd_class.pt")
        self.bd_train_dataset_wo_transform.set_state(state_dict)
        self.logger.debug(f"load backdoor train dataset from {global_path}/{self.args.save_folder}/bd_dataset"
                          f"/train_bd_class.pt")

        self.bd_test_dataset_wo_transform = prepare_bd_Dataset(
            deepcopy(self.clean_test_dataset_wo_transform), train=False)
        state_dict = torch.load(global_path + "/" + self.args.save_folder + "/bd_dataset/test_bd_class.pt")
        self.bd_test_dataset_wo_transform.set_state(state_dict)
        self.logger.debug(f"load backdoor test dataset from {global_path}/{self.args.save_folder}/bd_dataset"
                          f"/test_bd_class.pt")

        # load backdoor model
        self.device = torch.device(self.args.device)
        self.bd_model = get_model(self.args.model, self.args)
        self.bd_model.load_state_dict(torch.load(f"{global_path}/{self.args.save_folder}/parameter.pth"))
        self.bd_model = self.bd_model.to(self.device)
        self.logger.debug(f"load backdoor {self.args.model} model and its' parameters")


if __name__ == '__main__':
    a = defense("config/attack/badnet/bd_cifar10_resnet18.yaml")
    a.save_folder = "record/defense/abl/badnet_cifar10_resnet18"
    a.backdoor_prepare()
    for i in range(10000):
        x = a.bd_train_dataset_wo_transform[i]

