from torch.utils.data import DataLoader
import sys
import os
import argparse
import yaml
import logging
import time
import json
import csv
from copy import deepcopy
import numpy as np

sys.path.append('../')
from global_param import *

os.chdir(global_path)

from prototype.standard_framework import standard_framework
from utils.bd_transform_generate import peril_img_trans_generate, bd_attack_label_trans_generate
from utils.generate_poison_index import generate_poison_index_from_label_transform
from utils.backdoor_dataset import prepare_bd_Dataset
from datasets.get_dataset import get_dataset_and_transform
from datasets.ImageDataInfo import *
from models.get_model import get_model
from utils.path_check import path_check
from utils.fix_random import fix_random
from utils.args_check import args_check, get_lr_scheduler
from utils.load_save_model import load_checkpoint, save_checkpoint
from utils.dataset_wrapper import dataset_wrapper_with_transform


class Peril(standard_framework):
    def __init__(self, yaml_path):
        super(Peril, self).__init__()
        self.add_yaml_to_args(yaml_path)
        self.log_prepare()
        self.prepare()
        with open(self.yaml_path, 'r') as f:
            bd_defaults = yaml.safe_load(f)
            self.attack = bd_defaults['attack']
            self.patch_mask_path = bd_defaults['patch_mask_path']
            self.attack_label_trans = bd_defaults['attack_label_trans']
            self.attack_target = bd_defaults['attack_target']
            self.pratio = bd_defaults['pratio']
            self.clean_label_attack = bd_defaults['clean_label_attack']
            print("succeed to add backdoor config")

    def bd_dataset_prepare(self):
        ### get the backdoor transform on image
        self.bd_train_image_change, \
            self.bd_test_image_change = \
            peril_img_trans_generate(self.attack, self.patch_mask_path, self.args.image_size, self.args.dataset)

        ### get the backdoor transform on label
        self.bd_train_label_change = bd_attack_label_trans_generate(self.attack_label_trans, self.attack_target)
        self.bd_test_label_change = self.bd_train_label_change

        clean_train_dataset_label = np.array(
            [label for image, label, *other_info in self.clean_train_dataset_wo_transform])
        train_poison_index = generate_poison_index_from_label_transform(
            clean_train_dataset_label,
            label_transform=self.bd_train_label_change,
            target_label=self.attack_target,
            clean_label_attack=self.clean_label_attack,
            pratio=self.pratio,
            train=True)

        self.bd_train_dataset_wo_transform = prepare_bd_Dataset(
            deepcopy(self.clean_train_dataset_wo_transform),
            poison_indicator=train_poison_index,
            train=True,
            bd_image_pre_transform=self.bd_train_image_change,
            bd_label_pre_transform=self.bd_train_label_change,
            save_folder_path=global_path + "/" + self.args.save_folder
        )
        self.bd_train_dataset_wo_transform.save_class()
        bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            self.bd_train_dataset_wo_transform,
            self.clean_train_image_transform)
        self.train_dataloader = DataLoader(bd_train_dataset_with_transform,
                                           batch_size=self.args.batch_size,
                                           shuffle=True)
        self.logger.info("succeed to prepare backdoor train dataloader")

        clean_test_dataset_label = np.array(
            [label for image, label, *other_info in self.clean_test_dataset_wo_transform])
        test_poison_index_ASR = generate_poison_index_from_label_transform(
            clean_test_dataset_label,
            label_transform=self.bd_test_label_change,
            target_label=self.attack_target,
            clean_label_attack=self.clean_label_attack,
            pratio=self.pratio,
            train=False)
        self.bd_test_dataset_wo_transform = prepare_bd_Dataset(
            deepcopy(self.clean_test_dataset_wo_transform),
            poison_indicator=test_poison_index_ASR,
            train=False,
            bd_image_pre_transform=self.bd_test_image_change,
            bd_label_pre_transform=self.bd_test_label_change,
            save_folder_path=global_path + "/" + self.args.save_folder
        )
        self.bd_test_dataset_wo_transform.save_class()
        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            self.bd_test_dataset_wo_transform,
            self.clean_test_image_transform)

        self.test_dataloader_ASR = DataLoader(bd_test_dataset_with_transform,
                                              batch_size=self.args.batch_size,
                                              shuffle=False)
        self.logger.info("succeed to prepare backdoor test dataloader, all backdoor samples")

    def load_bd_dataset(self):
        self.bd_train_dataset_wo_transform = prepare_bd_Dataset(
            deepcopy(self.clean_train_dataset_wo_transform), train=True)
        state_dict = torch.load(global_path + "/" + self.args.save_folder + "/bd_dataset/train_bd_class.pt")
        self.bd_train_dataset_wo_transform.set_state(state_dict)
        self.logger.info(f"load backdoor train dataset from {global_path}/{self.args.save_folder}/bd_dataset"
                         f"/train_bd_class.pt")

        self.bd_test_dataset_wo_transform = prepare_bd_Dataset(
            deepcopy(self.clean_test_dataset_wo_transform), train=False)
        state_dict = torch.load(global_path + "/" + self.args.save_folder + "/bd_dataset/test_bd_class.pt")
        self.bd_test_dataset_wo_transform.set_state(state_dict)
        self.logger.info(
            f"load backdoor test dataset from {global_path}/{self.args.save_folder}/bd_dataset/test_bd_class.pt")

        bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            self.bd_train_dataset_wo_transform,
            self.clean_train_image_transform)
        self.train_dataloader = DataLoader(bd_train_dataset_with_transform,
                                           batch_size=self.args.batch_size,
                                           shuffle=True)

        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            self.bd_test_dataset_wo_transform,
            self.clean_test_image_transform)
        self.test_dataloader_ASR = DataLoader(bd_test_dataset_with_transform,
                                              batch_size=self.args.batch_size,
                                              shuffle=False)


if __name__ == '__main__':
    peril = Peril("config/attack/peril/bd_cifar10_WRN28.yaml")
    peril.bd_dataset_prepare()
    peril.train()
