import torch
from torch.utils.data import DataLoader
import sys
import os
import argparse
import yaml
import logging
import time
import json
import csv
import h5py
import numpy as np

sys.path.append('../')
from global_param import *

os.chdir(global_path)

from datasets.get_dataset import get_dataset_and_transform
from datasets.ImageDataInfo import *
from models.get_model import get_model
from utils.path_check import path_check
from utils.fix_random import fix_random
from utils.args_check import args_check, get_lr_scheduler, get_optimizer
from utils.load_save_model import load_checkpoint, save_checkpoint
from utils.dataset_wrapper import dataset_wrapper_with_transform


class standard_framework(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description=sys.argv[0])
        parser.add_argument("-d", "--dataset", type=str, help="which dataset to use")
        parser.add_argument('-m', '--model', type=str, help='which model to use')
        parser.add_argument('--drop_rate', type=float, help='drop rate in WRN')
        parser.add_argument('--pretrained', type=bool, help='whether or not to load pretrained parameters')
        parser.add_argument('-bs', '--batch_size', type=int)
        parser.add_argument("-dv", '--device', type=str)
        parser.add_argument('-op', '--optimizer', type=str, help='which optimizer to use')
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('-wd', '--weight_decay', type=float, help='weight decay of sgd')
        parser.add_argument('--random_seed', type=int, help='random_seed')
        parser.add_argument('-e', '--epochs', type=int)
        parser.add_argument('-lr', '--learning_rate', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='which lr_scheduler use for optimizer')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--steplr_T_max', type=int)
        parser.add_argument('--steplr_eta_min', type=float)
        parser.add_argument('--frequency_save', type=int, help=' frequency_save, 0 is never, save check_point')
        parser.add_argument('--path_checkpoint', type=str, help='path of checkpoint')
        parser.add_argument('--save_folder', type=str, help='folder name of the run file')
        parser.add_argument('-y', '--yaml_path', type=str, help='config file path')
        self.args = parser.parse_args()
        self.yaml_path = None

        self.attack = None

        self.logger = None
        self.clean_test_image_transform = None
        self.clean_test_dataset_wo_transform = None
        self.clean_train_image_transform = None
        self.clean_train_dataset_wo_transform = None

        # image: clean_train_dataset_wo_transform -bd_train_image_change-> bd_train_dataset_wo_transform
        # -clean_train_image_transform -> bd_train_dataset_with_transform
        # label: clean_train_dataset_wo_transform -bd_train_label_change -> bd_train_dataset_wo_transform
        self.bd_train_dataset_wo_transform = None
        self.bd_train_image_change = None
        self.bd_train_label_change = None
        self.bd_test_dataset_wo_transform = None
        self.bd_test_image_change = None
        self.bd_test_label_change = None

        self.train_dataloader = None  # dataloader used to train, in bd ,rewrite the class instance
        self.test_dataloader = None  # clean dataloader used to test, CA
        self.test_dataloader_ASR = None  # backdoor dataloader to test, ASR, all backdoor sample

        self.device = None
        self.model = None
        self.optimizer = None
        self.current_epoch = -1
        self.lr_scheduler = None
        self.criterion = None

    def add_yaml_to_args(self, yaml_path):
        if self.args.yaml_path is None:
            self.yaml_path = yaml_path
        else:
            self.yaml_path = self.args.yaml_path
        with open(self.yaml_path, 'r') as f:
            args_defaults = yaml.safe_load(f)
        self.args.__dict__.update(
            {k: v for k, v in args_defaults.items()
             if v is not None and k in self.args.__dict__ and self.args.__dict__[k] is None})
        print("succeed to add yaml to args")
        print(f'dataset:{self.args.dataset}, model:{self.args.model}')
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

    def log_prepare(self):
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        self.logger = logging.getLogger("mylogger")
        self.logger.setLevel(logging.DEBUG)
        # file Handler
        path_check(self.args.save_folder)
        fileHandler = logging.FileHandler(self.args.save_folder + '/debug.log')
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(logging.DEBUG)
        self.logger.addHandler(fileHandler)
        # consoleHandler
        consoleHandler = logging.StreamHandler(sys.stderr)
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(logging.INFO)
        self.logger.addHandler(consoleHandler)
        # overall logger level should <= min(handler) otherwise no log will be recorded.

        # disable other debug, since too many debug
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

        self.logger.debug("Only INFO or above level log will show in cmd. DEBUG level log only will show in log file.")

    def prepare(self):
        # check the args
        args_check(self.args)

        # set the random seed
        if self.args.random_seed is not None:
            fix_random(int(self.args.random_seed))
            self.logger.debug("fix randomness")

        self.clean_train_dataset_wo_transform, \
            self.clean_train_image_transform, \
            self.clean_test_dataset_wo_transform, \
            self.clean_test_image_transform = get_dataset_and_transform(self.args.dataset)
        self.logger.debug(f"succeed to load {self.args.dataset} clean dataset and transform")
        clean_train_dataset_with_transform = dataset_wrapper_with_transform(self.clean_train_dataset_wo_transform,
                                                                            self.clean_train_image_transform)
        clean_test_dataset_with_transform = dataset_wrapper_with_transform(self.clean_test_dataset_wo_transform,
                                                                           self.clean_test_image_transform)
        self.train_dataloader = DataLoader(clean_train_dataset_with_transform,
                                           batch_size=self.args.batch_size,
                                           shuffle=True)
        self.test_dataloader = DataLoader(clean_test_dataset_with_transform,
                                          batch_size=self.args.batch_size,
                                          shuffle=False)
        self.logger.debug(f"set the clean dataset as train and test dataloader")

        self.device = torch.device(self.args.device)
        self.logger.debug(f"We will use the device: {self.args.device}")

        self.model = get_model(self.args.model, self.args)
        self.logger.debug(f"We will use the model: {self.args.model}")

        self.optimizer = get_optimizer(self.model, self.args)
        self.logger.debug(f"We will use the optimizer: {self.args.optimizer}")

        # load check_point
        if self.args.path_checkpoint is not None:
            checkpoint = load_checkpoint(self.args.path_checkpoint)
            self.model.load_state_dict(checkpoint['net'])  # data in cpu
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.current_epoch = checkpoint['epoch']
            self.logger.info(f"succeed to load check_point into {self.args.model} and {self.args.optimizer}, "
                             f"and the epoch of checkpoint is {self.current_epoch}")
        # self.model = self.model.to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.logger.debug(f"We will use the cross_entropy loss")

        self.lr_scheduler = get_lr_scheduler(self.args, self.optimizer, self.current_epoch)
        self.logger.debug(f"We will use the lr_scheduler: {self.args.lr_scheduler}")

    def train_one_epoch(self):
        self.model.train()
        batch_loss_list = []
        start_time = time.time()
        for batch_idx, (x, label, *other_info) in enumerate(self.train_dataloader):
            x, label = x.to(self.device), label.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, label)
            batch_loss_list.append(x.shape[0] * loss.cpu().detach().item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        end_time = time.time()
        epoch_loss_avg = sum(batch_loss_list) / len(self.clean_train_dataset_wo_transform)
        self.logger.info(f"epoch:{self.current_epoch}, loss_avg: {epoch_loss_avg}, ----------------"
                         f"time:{end_time - start_time}")
        with open(self.args.save_folder + '/lr_loss.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow([self.current_epoch, self.optimizer.param_groups[0]['lr'], epoch_loss_avg])

    def test_traindata_loss(self):
        # test mode, compute the loss of clean and poisoned data in train dataloader
        bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            self.bd_train_dataset_wo_transform,
            self.clean_train_image_transform)
        bd_train_dataloader_wo_shuffle = DataLoader(bd_train_dataset_with_transform,
                                                    batch_size=self.args.batch_size,
                                                    shuffle=False)
        self.model.eval()
        clean_sample_loss = []
        clean_sample_num = 0
        backdoor_sample_loss = []
        backdoor_sample_num = 0
        original_index_all = []
        loss_all = []
        poison_or_not_all = []
        original_label_all = []
        poisoned_label_all = []
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            for batch_idx, (x, label, *other_info) in enumerate(bd_train_dataloader_wo_shuffle):
                if len(other_info) != 3:
                    return
                x, label = x.to(self.device), label.to(self.device)
                poison_or_not = other_info[1].float().to(self.device)
                logits = self.model(x)
                loss = criterion(logits, label)

                # save the loss per sample to hdf5
                original_index_all.append(other_info[0].cpu().detach())
                poison_or_not_all.append(other_info[1].cpu().detach())
                loss_all.append(loss.cpu().detach())
                original_label_all.append(other_info[2].cpu().detach())
                poisoned_label_all.append(label.cpu().detach())

                # compute the average loss of backdoor and clean samples
                clean_sample_num += torch.eq(poison_or_not, 0).sum().cpu().item()
                backdoor_sample_num += torch.eq(poison_or_not, 1).sum().cpu().item()
                backdoor_loss_temp = torch.dot(poison_or_not, loss)
                clean_loss_temp = torch.dot(1 - poison_or_not, loss)
                clean_sample_loss.append(clean_loss_temp.cpu().item())
                backdoor_sample_loss.append(backdoor_loss_temp.cpu().item())

        original_index_all = torch.cat(original_index_all, dim=0).cpu().detach()
        poison_or_not_all = torch.cat(poison_or_not_all, dim=0).cpu().detach()
        loss_all = torch.cat(loss_all, dim=0).cpu().detach()
        original_label_all = torch.cat(original_label_all, dim=0).cpu().detach()
        poisoned_label_all = torch.cat(poisoned_label_all, dim=0).cpu().detach()
        path_check(f"{self.args.save_folder}/loss_per_sample_epoch")
        h5_file = h5py.File(f"{self.args.save_folder}/loss_per_sample_epoch/loss_{self.current_epoch + 1}.hdf5",
                            "w")
        h5_file.create_dataset("original_index", data=original_index_all)
        h5_file.create_dataset("poison_or_not", data=poison_or_not_all)
        h5_file.create_dataset("loss_per_sample", data=loss_all)
        h5_file.create_dataset("original_label", data=original_label_all)
        h5_file.create_dataset("poisoned_label", data=poisoned_label_all)
        h5_file.close()

        clean_sample_loss_avg = np.sum(clean_sample_loss) / clean_sample_num
        backdoor_sample_loss_avg = np.sum(backdoor_sample_loss) / backdoor_sample_num
        with open(self.args.save_folder + '/train_loss_poison_and_not.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow([self.current_epoch + 1, clean_sample_loss_avg, backdoor_sample_loss_avg])

    def train_and_test_one_epoch(self):
        self.model.train()
        batch_train_loss_list = []
        start_time_train = time.time()
        for batch_idx, (x, label, *other_info) in enumerate(self.train_dataloader):
            x, label = x.to(self.device), label.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, label)
            batch_train_loss_list.append(x.shape[0] * loss.cpu().detach().item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        end_time_train = time.time()
        epoch_train_loss_avg = sum(batch_train_loss_list) / len(self.clean_train_dataset_wo_transform)
        self.logger.info(f"epoch:{self.current_epoch}, train_loss_avg: {epoch_train_loss_avg}, ----------------"
                         f"time:{end_time_train - start_time_train}")

        self.model.eval()
        batch_test_loss_list = []
        batch_test_acc_list = []
        start_time_test = time.time()
        for batch_idx, (x, label, *other_info) in enumerate(self.test_dataloader):
            x, label = x.to(self.device), label.to(self.device)
            logits = self.model(x)
            pred = logits.argmax(dim=1)
            loss = self.criterion(logits, label)
            batch_test_loss_list.append(x.shape[0] * loss.cpu().detach().item())
            batch_test_acc_list.append(pred.eq(label).sum().item())

        end_time_test = time.time()
        epoch_test_loss_avg = sum(batch_test_loss_list) / len(self.clean_test_dataset_wo_transform)
        epoch_acc = sum(batch_test_acc_list) / len(self.clean_test_dataset_wo_transform)
        self.logger.info(f"epoch:{self.current_epoch}, test_loss_avg: {epoch_test_loss_avg}, "
                         f"epoch_acc: {epoch_acc}-----------"
                         f"time:{end_time_test - start_time_test}")

        if self.test_dataloader_ASR is not None:
            self.model.eval()
            batch_test_loss_list = []
            batch_test_acc_list = []
            start_time_test = time.time()
            for batch_idx, (x, label, *other_info) in enumerate(self.test_dataloader_ASR):
                x, label = x.to(self.device), label.to(self.device)
                logits = self.model(x)
                pred = logits.argmax(dim=1)
                loss = self.criterion(logits, label)
                batch_test_loss_list.append(x.shape[0] * loss.cpu().detach().item())
                batch_test_acc_list.append(pred.eq(label).sum().item())

            end_time_test = time.time()
            epoch_test_bd_loss_avg = sum(batch_test_loss_list) / len(self.bd_test_dataset_wo_transform)
            epoch_ASR = sum(batch_test_acc_list) / len(self.bd_test_dataset_wo_transform)
            self.logger.info(f"epoch:{self.current_epoch}, bd_test_loss_avg: {epoch_test_bd_loss_avg}, "
                             f"epoch_ASR: {epoch_ASR}-----------"
                             f"time:{end_time_test - start_time_test}")

            with open(self.args.save_folder + '/lr_loss.csv', 'a') as file:
                writer = csv.writer(file)
                writer.writerow([self.current_epoch, self.optimizer.param_groups[0]['lr'],
                                 epoch_train_loss_avg, epoch_test_loss_avg, epoch_acc,
                                 epoch_test_bd_loss_avg, epoch_ASR])
            self.test_traindata_loss()

        else:
            with open(self.args.save_folder + '/lr_loss.csv', 'a') as file:
                writer = csv.writer(file)
                writer.writerow([self.current_epoch, self.optimizer.param_groups[0]['lr'],
                                 epoch_train_loss_avg, epoch_test_loss_avg, epoch_acc])

    def train(self):
        self.model = self.model.to(self.device)
        # save args.json
        path_check(self.args.save_folder)
        with open(self.args.save_folder + "/args.json", 'w', encoding='utf-8') as f:
            json.dump(self.args.__dict__, f, indent=4)
        self.logger.debug(f"save args_json to {self.args.save_folder}")

        if self.test_dataloader_ASR is not None:
            self.logger.info("Compute the average loss of clean and backdoor training samples")
            self.test_traindata_loss()

        self.logger.info("start to train")

        counter = 0  # when counter is 10, then save checkpoint
        path_check(self.args.save_folder + '/checkpoint')
        for self.current_epoch in range(self.current_epoch + 1, self.args.epochs):
            self.train_and_test_one_epoch()
            self.lr_scheduler.step()
            counter += 1
            if counter == self.args.frequency_save:
                checkpoint = {
                    'net': self.model.cpu().state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': self.current_epoch
                }
                self.model = self.model.to(self.device)
                save_checkpoint(self.current_epoch, self.args.save_folder + '/checkpoint', checkpoint)
                counter = 0

        torch.save(self.model.cpu().state_dict(), self.args.save_folder + "/parameter.pth")
        self.logger.info("finish training and save the parameters")


