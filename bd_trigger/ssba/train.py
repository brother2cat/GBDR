import argparse
import torch
import sys
import os
import yaml
import logging
import json
import csv
from tqdm import tqdm
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

sys.path.append('../../')
from global_param import *

os.chdir(global_path)
from datasets.get_dataset import get_dataset_and_transform
from utils.path_check import path_check
import bd_trigger.ssba.models as models
from datasets.ImageDataInfo import *
from utils.fix_random import fix_random
from utils.dataset_wrapper import dataset_wrapper_with_transform
import bd_trigger.ssba.generate_fingerprints as gf
from utils.load_save_model import load_checkpoint, save_checkpoint


class encoder_train:
    def __init__(self, yaml_path):
        parser = argparse.ArgumentParser(description=sys.argv[0])
        parser.add_argument("--description", default="Train the encoder-decoder of SSBA", type=str)
        parser.add_argument("-y", "--yaml_path", default=None, type=str)
        self.args = parser.parse_args()
        if self.args.yaml_path is not None:
            yaml_path = self.args.yaml_path
        with open(yaml_path, 'r') as f:
            args_defaults = yaml.safe_load(f)
        self.args.__dict__.update({k: v for k, v in args_defaults.items() if v is not None})
        print("succeed to add yaml to args")
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

        self.logger = None
        self.log_prepare()

        # set the random seed
        if self.args.random_seed is not None:
            fix_random(int(self.args.random_seed))
            self.logger.debug("fix randomness")

        self.clean_train_dataset_wo_transform, \
            self.clean_train_image_transform, \
            self.clean_test_dataset_wo_transform, \
            self.clean_test_image_transform = get_dataset_and_transform(self.args.dataset)
        self.logger.info(f"load the {self.args.dataset} clean dataset")

        clean_train_dataset_with_transform = dataset_wrapper_with_transform(self.clean_train_dataset_wo_transform,
                                                                            self.clean_train_image_transform)
        self.train_dataloader = DataLoader(clean_train_dataset_with_transform,
                                           batch_size=self.args.batch_size,
                                           shuffle=True)
        self.device = torch.device("cuda:0")
        self.encoder = models.StegaStampEncoder(
            self.args.image_size,
            self.args.channel,
            self.args.fingerprint_length,
            return_residual=0,
        )
        self.decoder = models.StegaStampDecoder(
            self.args.image_size,
            self.args.channel,
            self.args.fingerprint_length,
        )
        self.encoder, self.decoder = self.encoder.to(self.device), self.decoder.to(self.device)
        self.logger.info("load the encoder and decoder")

        self.optimizer = Adam(params=list(self.decoder.parameters()) + list(self.encoder.parameters()), lr=self.args.lr)

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

    def train(self):
        # save args.json
        path_check(self.args.save_folder)
        with open(self.args.save_folder + "/args.json", 'w', encoding='utf-8') as f:
            json.dump(self.args.__dict__, f, indent=4)
        self.logger.debug(f"save args_json to {self.args.save_folder}")

        global_step = 0
        steps_since_l2_loss_activated = -1

        self.encoder.train()
        self.decoder.train()
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        for i_epoch in range(self.args.num_epochs):
            bitwise_accuracy_epoch = []
            loss_epoch = []
            BCE_loss_epoch = []
            l2_loss_epoch = []
            for images, *other_info in tqdm(self.train_dataloader):
                global_step += 1
                batch_size = min(self.args.batch_size, images.size(0))

                fingerprints = gf.generate_random_fingerprints(self.args.fingerprint_length, batch_size)

                l2_loss_weight = min(
                    max(
                        0,
                        self.args.l2_loss_weight
                        * (steps_since_l2_loss_activated - self.args.l2_loss_await)
                        / self.args.l2_loss_ramp,
                    ),
                    self.args.l2_loss_weight,
                )

                BCE_loss_weight = self.args.BCE_loss_weight

                clean_images = images.to(self.device)
                fingerprints = fingerprints.to(self.device)

                fingerprinted_images = self.encoder(fingerprints, clean_images)
                residual = fingerprinted_images - clean_images

                decoder_output = self.decoder(fingerprinted_images)

                criterion = nn.MSELoss()
                l2_loss = criterion(fingerprinted_images, clean_images)

                criterion = nn.BCEWithLogitsLoss()
                BCE_loss = criterion(decoder_output.view(-1), fingerprints.view(-1))

                ##
                loss = l2_loss_weight * l2_loss + BCE_loss_weight * BCE_loss

                self.encoder.zero_grad()
                self.decoder.zero_grad()

                loss.backward()
                self.optimizer.step()

                fingerprints_predicted = (decoder_output > 0).float()
                bitwise_accuracy = 1.0 - torch.mean(torch.abs(fingerprints - fingerprints_predicted))

                if steps_since_l2_loss_activated == -1:
                    if bitwise_accuracy.item() > 0.9:
                        print(
                            "Current epoch: {}, Current global step: {}, Current bitwise acc: {}, Start to use l2 loss!".format(
                                i_epoch, global_step, bitwise_accuracy.item()))
                        steps_since_l2_loss_activated = 0
                else:
                    steps_since_l2_loss_activated += 1

                bitwise_accuracy_epoch.append(bitwise_accuracy.cpu().detach().item())
                loss_epoch.append(loss.cpu().detach().item())
                BCE_loss_epoch.append(BCE_loss.cpu().detach().item())
                l2_loss_epoch.append(l2_loss.cpu().detach().item())

            self.logger.info(f"epoch:{i_epoch}, loss:{np.mean(loss_epoch)}")
            with open(self.args.save_folder + '/loss.csv', 'a') as file:
                writer = csv.writer(file)
                writer.writerow([i_epoch,
                                 np.mean(bitwise_accuracy_epoch),
                                 np.mean(loss_epoch),
                                 np.mean(BCE_loss_epoch),
                                 np.mean(l2_loss_epoch)])

            if (i_epoch + 1) % 10000 == 0:
                checkpoint = {
                    'net':
                        {"encoder": self.encoder.cpu().state_dict(),
                         "decoder": self.decoder.cpu().state_dict()},
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': i_epoch
                }
                save_checkpoint(i_epoch, self.args.save_folder + '/checkpoints', checkpoint)
                self.logger.info(f"Save the {i_epoch} checkpoints")
                self.encoder = self.encoder.to(self.device)
                self.decoder = self.decoder.to(self.device)

        torch.save(self.encoder.cpu().state_dict(), self.args.save_folder + '/encoder.pth')
        torch.save(self.decoder.cpu().state_dict(), self.args.save_folder + '/decoder.pth')


if __name__ == '__main__':
    a = encoder_train("bd_trigger/ssba/config/train_param/cifar10.yaml")
    a.train()
