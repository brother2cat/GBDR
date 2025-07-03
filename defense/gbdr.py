import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
import sys
import os
import yaml
import json
import csv
import time
import copy
import h5py

sys.path.append('../')
from global_param import *

os.chdir(global_path)
from defense.base import defense
from models.get_model import get_model
from utils.path_check import path_check
from utils.args_check import get_optimizer, get_lr_scheduler
from utils.dataset_wrapper import dataset_wrapper_with_transform, dataset_wrapper_with_augment_and_transform
from utils.defense_utils.abl_utils import *
from utils.loss_per_example import compute_loss_per_example
from utils.h5py_data import *
from utils.defense_utils.our_utils import *
from utils.diff_utils.denoising_diffusion_pytorch import GaussianDiffusion, Unet


class our_defense(defense):
    def __init__(self, attack_yaml_path, defense_yaml_path, save_folder_path):
        super(our_defense, self).__init__(attack_yaml_path)
        if self.args.defense_yaml_path is None:
            pass
        else:
            defense_yaml_path = self.args.defense_yaml_path
        with open(defense_yaml_path, 'r') as f:
            df_defaults = yaml.safe_load(f)
        self.args.__dict__.update({k: v for k, v in df_defaults.items() if v is not None and
                                   (k not in self.args.__dict__ or self.args.__dict__[k] is None)})

        # load backdoor data
        if self.args.defense_save_folder_path is None:
            pass
        else:
            save_folder_path = self.args.defense_save_folder_path
        self.save_folder = save_folder_path

        self.backdoor_prepare()

        self.bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            self.bd_train_dataset_wo_transform,
            self.clean_train_image_transform)
        self.train_dataloader = None

        self.bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            self.bd_test_dataset_wo_transform,
            self.clean_test_image_transform)
        self.test_dataloader_ASR = DataLoader(self.bd_test_dataset_with_transform,
                                              batch_size=self.args.batch_size,
                                              shuffle=False)
        self.clean_test_dataset_with_transform = dataset_wrapper_with_transform(
            self.clean_test_dataset_wo_transform,
            self.clean_test_image_transform)
        self.test_dataloader_CA = DataLoader(self.clean_test_dataset_with_transform,
                                             batch_size=self.args.batch_size,
                                             shuffle=False)

        # We use a light model to process the backdoor dataset, so need to declare a new model, optimizer, lr_scheduler
        self.preprocess_model = get_model(self.args.preprocess_model, self.args).to(self.device)
        self.preprocess_optimizer = get_optimizer(self.preprocess_model, self.args)
        self.preprocess_lr_scheduler = get_lr_scheduler(self.args, self.preprocess_optimizer)

        # We use a target model to relabel the poisoned data that is classified by us
        self.relabel_model = get_model(self.args.model, self.args).to(self.device)
        self.relabel_optimizer = get_optimizer(self.relabel_model, self.args)
        self.relabel_lr_scheduler = get_lr_scheduler(self.args, self.relabel_optimizer)

        # We need to declare the target model and its' optimizer, lr_scheduler
        self.target_model = get_model(self.args.model, self.args).to(self.device)
        self.target_optimizer = get_optimizer(self.target_model, self.args)
        self.target_lr_scheduler = get_lr_scheduler(self.args, self.target_optimizer)

        # the model optimizer scheduler to train
        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.generate_model = Unet(dim=self.args.diff_dim,
                                   flash_attn=self.args.diff_flash_attn, channels=self.args.channel).to(self.device)
        self.diffusion_model = GaussianDiffusion(self.generate_model,
                                                 image_size=self.args.image_size if self.args.dataset != 'mnist' else 32,
                                                 objective=self.args.diff_objective).to(self.device)

        self.logger.debug("declare the preprocess model, optimizer, lr_scheduler")
        self.criterion = None
        self.current_epoch = None

        self.Db = None
        self.Dc = None
        self.Db_adv = None
        self.Dc_purity = None

        self.Dc_wo_transform = None
        self.Db_wo_transform = None

        self.Dc_and_recon_Db = None

        self.path_loss = None

    def test_traindata_loss(self):
        # test mode, compute the loss of every sample
        bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            self.bd_train_dataset_wo_transform,
            self.clean_train_image_transform)
        bd_train_dataloader_wo_shuffle = DataLoader(bd_train_dataset_with_transform,
                                                    batch_size=self.args.batch_size,
                                                    shuffle=False)
        self.preprocess_model.eval()
        self.preprocess_model = self.preprocess_model.to(self.device)
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
                logits = self.preprocess_model(x)
                loss = criterion(logits, label)

                # save the loss per sample to hdf5
                original_index_all.append(other_info[0].cpu().detach())
                poison_or_not_all.append(other_info[1].cpu().detach())
                loss_all.append(loss.cpu().detach())
                original_label_all.append(other_info[2].cpu().detach())
                poisoned_label_all.append(label.cpu().detach())

        original_index_all = torch.cat(original_index_all, dim=0).cpu().detach()
        poison_or_not_all = torch.cat(poison_or_not_all, dim=0).cpu().detach()
        loss_all = torch.cat(loss_all, dim=0).cpu().detach()
        original_label_all = torch.cat(original_label_all, dim=0).cpu().detach()
        poisoned_label_all = torch.cat(poisoned_label_all, dim=0).cpu().detach()
        path_check(f"{self.save_folder}/loss_per_sample_epoch")
        self.logger.debug(f"Save {self.save_folder}/loss_per_sample_epoch/loss_{str(self.current_epoch + 1)}.hdf5")
        h5_file = h5py.File(f"{self.save_folder}/loss_per_sample_epoch/loss_{str(self.current_epoch + 1)}.hdf5",
                            "w")
        h5_file.create_dataset("original_index", data=original_index_all)
        h5_file.create_dataset("poison_or_not", data=poison_or_not_all)
        h5_file.create_dataset("loss_per_sample", data=loss_all)
        h5_file.create_dataset("original_label", data=original_label_all)
        h5_file.create_dataset("poisoned_label", data=poisoned_label_all)
        h5_file.close()

    def train_and_test_one_epoch(self):
        # self.criterion is for train,
        self.model.train()
        self.model = self.model.to(self.device)
        batch_train_loss_list = []
        num_train_data = 0
        start_time_train = time.time()
        for batch_idx, (x, label, *other_info) in enumerate(self.train_dataloader):
            x, label = x.to(self.device), label.to(self.device)
            logits = self.model(x)
            if isinstance(self.criterion, ADV_loss):
                mask = other_info[3].float().to(self.device)
                loss = self.criterion(logits, mask, label)
            else:
                loss = self.criterion(logits, label)
            num_train_data += x.shape[0]
            batch_train_loss_list.append(x.shape[0] * loss.cpu().detach().item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        end_time_train = time.time()
        epoch_train_loss_avg = sum(batch_train_loss_list) / num_train_data
        self.logger.info(f"epoch:{self.current_epoch}, train_loss_avg: {epoch_train_loss_avg}, ----------------"
                         f"time:{end_time_train - start_time_train}")

        self.model.eval()
        batch_test_loss_list = []
        batch_test_acc_list = []
        criterion = nn.CrossEntropyLoss()
        start_time_test = time.time()
        with torch.no_grad():
            for batch_idx, (x, label, *other_info) in enumerate(self.test_dataloader_CA):
                x, label = x.to(self.device), label.to(self.device)
                logits = self.model(x)
                pred = logits.argmax(dim=1)
                loss = criterion(logits, label)
                batch_test_loss_list.append(x.shape[0] * loss.cpu().detach().item())
                batch_test_acc_list.append(pred.eq(label).sum().item())

        end_time_test = time.time()
        epoch_test_loss_avg = sum(batch_test_loss_list) / len(self.clean_test_dataset_with_transform)
        epoch_acc = sum(batch_test_acc_list) / len(self.clean_test_dataset_with_transform)
        self.logger.info(f"epoch:{self.current_epoch}, CA_loss_avg: {epoch_test_loss_avg}, "
                         f"epoch_CA: {epoch_acc}-----------"
                         f"time:{end_time_test - start_time_test}")

        self.model.eval()
        batch_test_loss_list = []
        batch_test_acc_list = []
        criterion = nn.CrossEntropyLoss()
        start_time_test = time.time()
        with torch.no_grad():
            for batch_idx, (x, label, *other_info) in enumerate(self.test_dataloader_ASR):
                x, label = x.to(self.device), label.to(self.device)
                logits = self.model(x)
                pred = logits.argmax(dim=1)
                loss = criterion(logits, label)
                batch_test_loss_list.append(x.shape[0] * loss.cpu().detach().item())
                batch_test_acc_list.append(pred.eq(label).sum().item())

        end_time_test = time.time()
        epoch_test_bd_loss_avg = sum(batch_test_loss_list) / len(self.bd_test_dataset_with_transform)
        epoch_ASR = sum(batch_test_acc_list) / len(self.bd_test_dataset_with_transform)
        self.logger.info(f"epoch:{self.current_epoch}, bd_test_loss_avg: {epoch_test_bd_loss_avg}, "
                         f"epoch_ASR: {epoch_ASR}-----------"
                         f"time:{end_time_test - start_time_test}")

        self.logger.debug(f"Save {self.save_folder}{self.path_loss}")
        with open(self.save_folder + self.path_loss, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([self.current_epoch, self.optimizer.param_groups[0]['lr'],
                             epoch_train_loss_avg, epoch_test_loss_avg, epoch_acc,
                             epoch_test_bd_loss_avg, epoch_ASR])

    def pre_train(self):
        self.logger.info("Begin the first stage of Backdoor classification - (1) Preprocess train")
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.logger.info("set the loss function with CrossEntropy Loss")
        self.model = self.preprocess_model
        self.optimizer = self.preprocess_optimizer
        self.scheduler = self.preprocess_lr_scheduler
        self.path_loss = '/pre_process_lr_loss.csv'

        path_check(self.save_folder)
        with open(self.save_folder + "/args.json", 'w', encoding='utf-8') as f:
            json.dump(self.args.__dict__, f, indent=4)
        self.logger.debug(f"save args_json to {self.save_folder}")

        self.train_dataloader = DataLoader(self.bd_train_dataset_with_transform,
                                           batch_size=self.args.batch_size,
                                           shuffle=True)
        self.logger.info("Set the train_dataloader with bd_train_dataset_with_transform")

        # self.preprocess_model = self.preprocess_model.to(self.device)

        # Test the loss of every sample, Before training
        self.logger.info("Compute the loss of training samples BEFORE training")
        self.current_epoch = -1
        self.test_traindata_loss()

        for self.current_epoch in range(self.args.preprocess_epochs):
            self.train_and_test_one_epoch()
            self.test_traindata_loss()
            self.scheduler.step()
        self.logger.debug(f"Save {self.save_folder + '/parameter_pretrain.pth'}")
        torch.save(self.preprocess_model.cpu().state_dict(), self.save_folder + "/parameter_pretrain.pth")

    def load_pre_train(self):
        self.preprocess_model = self.preprocess_model.cpu()
        self.preprocess_model.load_state_dict(torch.load(self.save_folder + "/parameter_pretrain.pth"))
        self.preprocess_model = self.preprocess_model.to(self.device)

    def load_diffusion_model(self):
        self.diffusion_model = self.diffusion_model.cpu()
        self.diffusion_model.load_state_dict(torch.load(self.save_folder + "/diffusion/parameter_diff.pth"))
        self.diffusion_model = self.diffusion_model.to(self.device)

    def load_relabel_train(self):
        self.relabel_model = self.relabel_model.cpu()
        self.relabel_model.load_state_dict(torch.load(self.save_folder + "/relabel/parameter_clean_train.pth"))
        self.relabel_model = self.relabel_model.to(self.device)

    def load_relabel_final(self):
        self.relabel_model = self.relabel_model.cpu()
        self.relabel_model.load_state_dict(torch.load(self.save_folder + "/relabel/parameter_final_finetune.pth"))
        self.relabel_model = self.relabel_model.to(self.device)

    def isolate_dataset(self):
        self.logger.info("Begin the first stage of Backdoor classification - (2) Isolated data")
        loss_per_example_dict = compute_loss_per_example(self.preprocess_model,
                                                         self.bd_train_dataset_with_transform,
                                                         self.device)

        # sort according to the value
        loss_per_example_dict = dict(sorted(loss_per_example_dict.items(), key=lambda x: x[1]))
        num_poisoned = int(self.args.isolation_ratio * len(self.bd_train_dataset_with_transform))
        poisoned_index_list = list(loss_per_example_dict.keys())[:num_poisoned]
        clean_index_list = list(loss_per_example_dict.keys())[num_poisoned:]

        # get adv sample
        num_adv_poisoned = int(self.args.adv_isolation_ratio * len(self.bd_train_dataset_with_transform))
        adv_index_list = list(loss_per_example_dict.keys())[:num_adv_poisoned]

        # get clean purity sample
        num_clean_purity = int(self.args.clean_purity_isolation_ratio * len(self.bd_train_dataset_with_transform))
        clean_purity_index_list = list(loss_per_example_dict.keys())[
                                  int(len(self.bd_train_dataset_with_transform) - num_clean_purity):]

        self.Db = copy.deepcopy(self.bd_train_dataset_with_transform)
        self.Db.subset(poisoned_index_list)
        self.Dc = copy.deepcopy(self.bd_train_dataset_with_transform)
        self.Dc.subset(clean_index_list)
        self.Db_adv = copy.deepcopy(self.bd_train_dataset_with_transform)
        self.Db_adv.subset(adv_index_list)
        self.Dc_purity = copy.deepcopy(self.bd_train_dataset_with_transform)
        self.Dc_purity.subset(clean_purity_index_list)

        pil2tensor = transforms.ToTensor()
        self.Dc_wo_transform = dataset_wrapper_with_transform(self.bd_train_dataset_wo_transform, pil2tensor)
        self.Dc_wo_transform.subset(clean_index_list)
        self.Db_wo_transform = dataset_wrapper_with_transform(self.bd_train_dataset_wo_transform, pil2tensor)
        self.Db_wo_transform.subset(poisoned_index_list)

        TP, FP, FN, TN = compute_classification_acc(self.Db, self.Dc)
        total_num = len(self.bd_train_dataset_with_transform)
        acc = (TP + TN) / total_num
        precision = TP / (TP + FP)
        TPR = TP / (TP + FN)
        f1 = (2 * precision * TPR) / (precision + TPR)
        self.logger.info(f"Classification of poisoned samples using Loss Value, "
                         f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}, acc:{acc}, f1:{f1}")

        TP, FP, FN, TN = compute_classification_acc(self.Db_adv)
        self.logger.info(f"High purity backdoor samples, TP: {TP}, FP: {FP}")

        TP, FP, FN, TN = compute_classification_acc(None, self.Dc_purity)
        self.logger.info(f"High purity clean samples, FN: {FN}, TN: {TN}")

    def diffusion_model_train(self):
        self.logger.info("Begin the Second stage of Dataset Purification - (1) Diffusion Model Train")
        self.train_dataloader = DataLoader(self.Dc_wo_transform, batch_size=32, shuffle=True)
        self.logger.info("Set the train dataloader in diffusion with Dc_wo_transform")

        self.optimizer = torch.optim.AdamW(self.diffusion_model.parameters(),
                                           lr=1.e-4, betas=(0.9, 0.999), weight_decay=1.e-6)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=20, eta_min=1.e-5, last_epoch=-1)
        self.diffusion_model = self.diffusion_model.to(self.device)
        self.diffusion_model.train()
        for epoch in range(self.args.diff_epochs):
            loss_total = 0
            start_time = time.time()
            for batch_idx, (x, *_) in enumerate(self.train_dataloader):
                if self.args.dataset == 'mnist':
                    x = pad28to32with0(x)
                x = x.to(self.device)
                loss = self.diffusion_model(x)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_total += loss.cpu().detach().item() * x.shape[0]

            epoch_time = time.time() - start_time
            loss_avg = loss_total / len(self.Dc_wo_transform)
            self.logger.info(f"epoch:{epoch}, loss_avg: {loss_avg}, ----------------time:{epoch_time}")
            path_check(self.save_folder + '/diffusion')
            with open(self.save_folder + '/diffusion/lr_loss.csv', 'a') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, self.optimizer.param_groups[0]['lr'], loss_avg])
            self.scheduler.step()
        torch.save(self.diffusion_model.cpu().state_dict(), self.save_folder + '/diffusion/parameter_diff.pth')
        self.logger.info('Finish Diffusion Model Training')

    def diffusion_model_recon(self):
        self.logger.info("Begin the Second stage of Dataset Purification-(2) Diffusion Model Reconstruct Backdoor Data")
        recon_batch_size = 256
        recon_dataloader = DataLoader(self.Db_wo_transform, batch_size=recon_batch_size, shuffle=False)

        self.diffusion_model = self.diffusion_model.to(self.device)
        self.diffusion_model.eval()
        denoise_images_all = []
        other_info_all = [[] for i in range(4)]
        for batch_id, (image, *other_info) in enumerate(recon_dataloader):
            if self.args.dataset == 'mnist':
                image = pad28to32with0(image)
            t = torch.LongTensor([self.args.diff_steps_forward for i in range(image.shape[0])]).to(self.device)
            image = image.to(self.device)
            image = self.diffusion_model.q_sample(image, t)
            denoise_image = self.diffusion_model. \
                p_sample_loop_specific_image(image, steps_reverse=self.args.diff_steps_reverse,
                                             return_all_timesteps=False)

            if self.args.dataset == 'mnist':
                denoise_image = crop32to28(denoise_image)
            denoise_images_all.append(denoise_image.detach().cpu())
            for i in range(4):
                other_info_all[i].append(other_info[i])

        denoise_images_all = torch.cat(denoise_images_all, dim=0)
        for i in range(4):
            other_info_all[i] = torch.cat(other_info_all[i], dim=0)

        path_check(f"{self.save_folder}/diffusion")
        h5_file = h5py.File(f"{self.save_folder}/diffusion/recon_data_bd.hdf5", "w")
        h5_file.create_dataset("image", data=denoise_images_all)
        for i in range(4):
            h5_file.create_dataset(f"info{i}", data=other_info_all[i])
        h5_file.close()
        self.logger.info(f"Save the recon Db train dataset in {self.save_folder}/diffusion/recon_data_bd.hdf5")

    def relabel_train(self):
        self.logger.info("Begin the Second stage of Dataset Purification (3) Label Purification -- ReLabel Model Train")
        self.model = self.relabel_model.to(self.device)
        self.optimizer = self.relabel_optimizer
        self.scheduler = self.relabel_lr_scheduler
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        path_check(f"{self.save_folder}/relabel")
        self.path_loss = '/relabel/clean_train_lr_loss.csv'
        self.logger.info("set the model with relabel model, loss function with CrossEntropy")

        self.train_dataloader = DataLoader(self.Dc,
                                           batch_size=self.args.batch_size,
                                           shuffle=True)
        self.logger.info("Set the train_dataloader with Dc without any data augment")

        for self.current_epoch in range(self.args.epochs):
            self.train_and_test_one_epoch()
            self.scheduler.step()
        torch.save(self.relabel_model.cpu().state_dict(), self.save_folder + "/relabel/parameter_clean_train.pth")

    def relabel_adversarial_finetune(self, num_iter):
        self.logger.info("Begin the Second stage of Dataset Purification (3) Label Purification -- ReLabel Model "
                         "Adversarial Finetune")
        self.criterion = Adv_finetune_loss().to(self.device)
        self.logger.info("set the loss function with ADV Finetune CrossEntropy")

        self.train_dataloader = DataLoader(self.Db_adv,
                                           batch_size=self.args.batch_size,
                                           shuffle=True)
        self.logger.info("Set the train_dataloader with Db_adv")

        self.model = self.relabel_model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.relabel_model.parameters(),
                                           lr=self.args.lr_adv_finetune,
                                           betas=(0.9, 0.999),
                                           weight_decay=1e-06)
        self.path_loss = '/relabel/clean_train_lr_loss.csv'

        for self.current_epoch in range(self.args.adv_finetune_epochs):
            scheduler_adv_finetune(self.optimizer, self.current_epoch, self.args, self.logger, num_iter)
            self.train_and_test_one_epoch()
        torch.save(self.relabel_model.cpu().state_dict(), self.save_folder + "/relabel/parameter_adv_finetune.pth")

    def relabel_clean_purity_finetune(self, num_iter):
        self.logger.info("Begin the Second stage of Dataset Purification (3) Label Purification -- ReLabel Model "
                         "Clean Purity Finetune")
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.logger.info("set the loss function with CrossEntropy")

        self.train_dataloader = DataLoader(self.Dc_purity,
                                           batch_size=self.args.batch_size,
                                           shuffle=True)
        self.logger.info("Set the train_dataloader with Dc_purity")

        self.model = self.relabel_model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.relabel_model.parameters(),
                                           lr=self.args.lr_clean_purity_finetune,
                                           betas=(0.9, 0.999),
                                           weight_decay=1e-06)
        self.path_loss = '/relabel/clean_train_lr_loss.csv'

        for self.current_epoch in range(self.args.clean_purity_finetune_epochs):
            scheduler_clean_purity_finetune(self.optimizer, self.current_epoch, self.args, self.logger, num_iter)
            self.train_and_test_one_epoch()
        torch.save(self.relabel_model.cpu().state_dict(),
                   self.save_folder + "/relabel/parameter_clean_purity_finetune.pth")

    def relabel_finetune_iteration(self):
        self.logger.info(f"Begin the Second stage of Dataset Purification (3) Label Purification -- ReLabel Model "
                         f"Iterative Bidirectional Fine-tuning")
        for num_iter in range(self.args.num_iteration_finetune):
            self.logger.info(f"Begin the {num_iter}-th Bidirectional Fine-tuning")
            self.relabel_adversarial_finetune(num_iter)
            self.relabel_clean_purity_finetune(num_iter)
        torch.save(self.relabel_model.cpu().state_dict(),
                   self.save_folder + "/relabel/parameter_final_finetune.pth")
        self.logger.info(f"Save the final finetune Relabel Model in "
                         f"{self.save_folder}/relabel/parameter_final_finetune.pth")

    def relabelDc_reconDb_target_model_train(self):
        self.logger.info("Begin the final step - Standard Train with Relabel Dc and Relabel recon Db")
        self.model = self.target_model.to(self.device)
        self.optimizer = self.target_optimizer
        self.scheduler = self.target_lr_scheduler
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.path_loss = '/RelabelDc_ReconDb_train_lr_loss.csv'
        self.logger.info("Set the model with target model, loss function with CrossEntropy")

        Db_recon = h5py_recon_relabel_image_other_info(
            f"{self.save_folder}/relabel/recon_data_relabel_trans_bd.hdf5")
        Relabel_Dc = h5py_recon_relabel_image_other_info(
            f"{self.save_folder}/relabel/relabel_trans_clean.hdf5")

        self.Dc_and_recon_Db = Mix_two_Dataset(Relabel_Dc, Db_recon)
        self.train_dataloader = DataLoader(self.Dc_and_recon_Db,
                                           batch_size=self.args.batch_size,
                                           shuffle=True)
        self.logger.info("Set the train_dataloader with Relabel Dc and Relabel Db_recon")

        for self.current_epoch in range(self.args.epochs):
            self.train_and_test_one_epoch()
            self.scheduler.step()
        torch.save(self.target_model.cpu().state_dict(), self.save_folder + "/parameter_target_model_standard.pth")

    def label_purify(self):
        self.logger.info("Begin the Second stage of Dataset Purification - (4) Label Purification ---- Save dataset")
        self.model = self.relabel_model.to(self.device)
        Db_recon = h5py_recon_image_other_info(f"{self.save_folder}/diffusion/recon_data_bd.hdf5")
        Db_recon = dataset_wrapper_with_augment_and_transform(Db_recon,
                                                              None,
                                                              self.clean_train_image_transform)
        relabel_dataloader = DataLoader(Db_recon,
                                        batch_size=self.args.batch_size,
                                        shuffle=False)
        self.model.eval()
        image_all = []
        relabel_all = []
        other_info_all = [[] for i in range(3)]
        with torch.no_grad():
            for batch_id, (image, label, *other_info) in enumerate(relabel_dataloader):
                image = image.to(self.device)
                logits = self.model(image)
                pred = logits.argmax(dim=1)

                image_all.append(image.cpu())
                relabel_all.append(pred.cpu())
                for i in range(3):
                    other_info_all[i].append(other_info[i])
        image_all = torch.cat(image_all, dim=0)
        relabel_all = torch.cat(relabel_all, dim=0)
        for i in range(3):
            other_info_all[i] = torch.cat(other_info_all[i], dim=0)

        path_check(f"{self.save_folder}/relabel")
        h5_file = h5py.File(f"{self.save_folder}/relabel/recon_data_relabel_trans_bd.hdf5", "w")
        h5_file.create_dataset("image", data=image_all)
        h5_file.create_dataset("relabel", data=relabel_all)
        for i in range(3):
            h5_file.create_dataset(f"info{i}", data=other_info_all[i])
        h5_file.close()
        self.logger.info(f"Save the recon relabel transform Db train dataset in"
                         f" {self.save_folder}/relabel/recon_data_relabel_trans_bd.hdf5")

        ############################################################################################

        self.logger.debug(f"Begin to relabel the Dc with relabel model")
        # This is because when we directly train model on Dc, the model will be backdoored.
        relabel_dataloader = DataLoader(self.Dc,
                                        batch_size=self.args.batch_size,
                                        shuffle=False)
        self.model.eval()
        image_all = []
        relabel_all = []
        other_info_all = [[] for i in range(3)]
        with torch.no_grad():
            for batch_id, (image, label, *other_info) in enumerate(relabel_dataloader):
                image = image.to(self.device)
                logits = self.model(image)
                pred = logits.argmax(dim=1)

                image_all.append(image.cpu())
                relabel_all.append(pred.cpu())
                for i in range(3):
                    other_info_all[i].append(other_info[i])
        image_all = torch.cat(image_all, dim=0)
        relabel_all = torch.cat(relabel_all, dim=0)
        for i in range(3):
            other_info_all[i] = torch.cat(other_info_all[i], dim=0)

        path_check(f"{self.save_folder}/relabel")
        h5_file = h5py.File(f"{self.save_folder}/relabel/relabel_trans_clean.hdf5", "w")
        h5_file.create_dataset("image", data=image_all)
        h5_file.create_dataset("relabel", data=relabel_all)
        for i in range(3):
            h5_file.create_dataset(f"info{i}", data=other_info_all[i])
        h5_file.close()
        self.logger.info(f"Save the recon relabel transform Dc train dataset in"
                         f" {self.save_folder}/relabel/relabel_trans_clean.hdf5")


if __name__ == '__main__':
    a = our_defense(attack_yaml_path="config/attack/badnet/bd_cifar10_WRN28.yaml",
                    defense_yaml_path="config/defense/gbdr/cifar10.yaml",
                    save_folder_path="record/defense/gbdr/badnet_cifar10_WRN28")
    a.pre_train()
    a.isolate_dataset()
    a.diffusion_model_train()
    a.diffusion_model_recon()
    a.relabel_train()
    a.relabel_finetune_iteration()
    a.label_purify()
    a.relabelDc_reconDb_target_model_train()
