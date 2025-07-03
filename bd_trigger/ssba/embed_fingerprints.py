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
import bchlib
import h5py
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
from datasets.de_normalize import de_normalize
from utils.fix_random import fix_random
from utils.dataset_wrapper import dataset_wrapper_with_transform
import bd_trigger.ssba.generate_fingerprints as gf
from utils.load_save_model import load_checkpoint, save_checkpoint


class embed_fingerprints:
    def __init__(self, yaml_path):
        parser = argparse.ArgumentParser(description=sys.argv[0])
        parser.add_argument("--description",
                            default="Generate the poisoned dataset by encoder-decoder of SSBA", type=str)
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

        # set the random seed
        if self.args.random_seed is not None:
            fix_random(int(self.args.random_seed))
            print("fix randomness")

        self.device = torch.device("cuda:0")
        self.encoder = None
        self.decoder = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.load_model_dataloader()

    def load_model_dataloader(self):
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
        self.encoder.load_state_dict(torch.load(self.args.encoder_path))
        self.decoder.load_state_dict(torch.load(self.args.decoder_path))

        clean_train_dataset_wo_transform, \
            clean_train_image_transform, \
            clean_test_dataset_wo_transform, \
            clean_test_image_transform = get_dataset_and_transform(self.args.dataset)
        print(f"load the {self.args.dataset} clean dataset")

        clean_train_dataset_with_transform = dataset_wrapper_with_transform(clean_train_dataset_wo_transform,
                                                                            clean_test_image_transform)
        clean_test_dataset_with_transform = dataset_wrapper_with_transform(clean_test_dataset_wo_transform,
                                                                           clean_test_image_transform)
        self.train_dataloader = DataLoader(clean_train_dataset_with_transform,
                                           batch_size=self.args.batch_size,
                                           shuffle=False)
        self.test_dataloader = DataLoader(clean_test_dataset_with_transform,
                                          batch_size=self.args.batch_size,
                                          shuffle=False)
        self.embed(train=True)
        self.embed(train=False)

    def embed(self, train: bool):
        # train if True, self.train_dataloader
        # train if False, self.test_dataloader
        all_fingerprinted_images = []
        all_fingerprints = []
        all_code = []
        all_label = torch.tensor([])
        all_initial_image = torch.tensor([])
        BCH_POLYNOMIAL = 137

        FINGERPRINT_SIZE = self.encoder.secret_dense.weight.shape[-1]
        fingerprints = gf.generate_fingerprints(
            type=self.args.encode_method,
            batch_size=self.args.batch_size,
            fingerprint_size=FINGERPRINT_SIZE,
            secret=self.args.secret,
            seed=self.args.seed,
            diff_bits=self.args.diff_bits,
            manual_str=None,
            proportion=self.args.proportion,
            identical=self.args.identical_fingerprints)
        fingerprints = fingerprints.to(self.device)

        torch.manual_seed(self.args.seed)
        self.encoder.eval()
        self.decoder.eval()
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        # train dataset
        bitwise_accuracy = 0
        correct = 0
        with torch.no_grad():
            if train:
                print(f"Generate the backdoor data of train dataset")
            else:
                print(f"Generate the backdoor data of test dataset")
            for images, label in tqdm(self.train_dataloader if train else self.test_dataloader):
                all_initial_image = torch.cat([all_initial_image, images.cpu().detach()])
                all_label = torch.cat([all_label, label.cpu().detach()])

                images = images.to(self.device)
                fingerprinted_images = self.encoder(fingerprints[: images.size(0)], images)

                all_fingerprinted_images.append(fingerprinted_images.detach().cpu())
                all_fingerprints.append(fingerprints[: images.size(0)].detach().cpu())

                if self.args.check:
                    detected_fingerprints = self.decoder(fingerprinted_images)
                    detected_fingerprints = (detected_fingerprints > 0).long()
                    bitwise_accuracy += (detected_fingerprints[: images.size(0)].detach() == fingerprints[: images.size(
                        0)]).float().mean(dim=1).sum().item()
                    if self.args.encode_method == 'bch':
                        for sec in detected_fingerprints:
                            sec = np.array(sec.cpu())
                            if FINGERPRINT_SIZE == 100:
                                BCH_BITS = 5
                                bch = bchlib.BCH(BCH_BITS, BCH_POLYNOMIAL)
                                packet_binary = "".join([str(int(bit)) for bit in sec[:96]])
                            elif FINGERPRINT_SIZE == 50:
                                BCH_BITS = 2
                                bch = bchlib.BCH(BCH_BITS, BCH_POLYNOMIAL)
                                packet_binary = "".join([str(int(bit)) for bit in sec[:48]])
                            packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
                            packet = bytearray(packet)
                            data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]
                            bitflips = bch.decode(data, ecc)
                            if bitflips != -1:
                                try:
                                    correct += 1
                                    code = data.decode("utf-8")
                                    all_code.append(code)
                                    continue
                                except:
                                    all_code.append("Something went wrong")
                                    continue
                            all_code.append('Failed to decode')

        all_fingerprinted_images = torch.cat(all_fingerprinted_images, dim=0).cpu().detach()
        all_fingerprints = torch.cat(all_fingerprints, dim=0).cpu().detach()

        all_fingerprinted_images = de_normalize(self.args.dataset, all_fingerprinted_images)

        if train:
            h5_file = h5py.File(f"{self.args.save_folder}/train_bd.hdf5", "w")
        else:
            h5_file = h5py.File(f"{self.args.save_folder}/test_bd.hdf5", "w")
        h5_file.create_dataset("fingerprinted_images", data=all_fingerprinted_images)
        h5_file.create_dataset("label", data=all_label)
        h5_file.create_dataset("initial_images", data=all_initial_image)
        h5_file.close()

        if self.args.check:
            bitwise_accuracy = bitwise_accuracy / len(all_fingerprints)
            if self.args.encode_method == 'bch':
                decode_acc = correct / len(all_fingerprints)
                print(f"Decoding accuracy on fingerprinted images: {decode_acc}")
            print(f"Bitwise accuracy on fingerprinted images: {bitwise_accuracy}")


def main():
    a = embed_fingerprints("bd_trigger/ssba/config/test_param/cifar10.yaml")


if __name__ == '__main__':
    main()
