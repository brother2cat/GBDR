
import sys, logging
import os

sys.path.append('../')
from global_param import *

os.chdir(global_path)
import imageio
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from utils.bd_img_transform.patch import AddMaskPatchTrigger, SimpleAdditiveTrigger, AddPatchTrigger
from utils.bd_img_transform.SSBA import SSBA_attack_replace_version
from utils.bd_img_transform.ftrojan import ftrojann_version
from utils.bd_img_transform.color import Color_backdoor_shift
from utils.bd_label_transform.backdoor_label_transform import *
from torchvision.transforms import Resize


class general_compose(object):
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, img, *args, **kwargs):
        for transform, if_all in self.transform_list:
            if if_all == False:
                img = transform(img)
            else:
                img = transform(img, *args, **kwargs)
        return img


class convertNumpyArrayToFloat32(object):
    def __init__(self):
        pass

    def __call__(self, np_img_float32):
        return np_img_float32.astype(np.float32)


npToFloat32 = convertNumpyArrayToFloat32()


class convertFloat32NumpyArrayToUint8(object):
    def __init__(self):
        pass

    def __call__(self, np_img_float32):
        return np.clip(np_img_float32 * 255, 0, 255).astype(np.uint8)


npFloat32ToUint8 = convertFloat32NumpyArrayToUint8()


class clipAndConvertNumpyArrayToUint8(object):
    def __init__(self):
        pass

    def __call__(self, np_img_float32):
        return np.clip(np_img_float32, 0, 255).astype(np.uint8)


npClipAndToUint8 = clipAndConvertNumpyArrayToUint8()


def badnet_img_trans_generate(attack_name, patch_mask_path, image_size, dataset):
    if attack_name != "badnet":
        raise ValueError(f"We are conducting badnet backdoor attack but not {attack_name}")
    trans = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        np.array,
    ])

    if dataset == "mnist":
        bd_transform = AddMaskPatchTrigger(
            trans(Image.open(patch_mask_path).convert("L")),
        )
    else:
        bd_transform = AddMaskPatchTrigger(
            trans(Image.open(patch_mask_path)),
        )

    train_bd_transform = general_compose([
        (transforms.Resize((image_size, image_size)), False),
        (np.array, False),
        (bd_transform, True),
        (npClipAndToUint8, False),
    ])

    test_bd_transform = general_compose([
        (transforms.Resize((image_size, image_size)), False),
        (np.array, False),
        (bd_transform, True),
        (npClipAndToUint8, False),
    ])
    return train_bd_transform, test_bd_transform


def ssba_img_trans_generate(attack_name, path_replace_image, image_size, dataset):
    if attack_name != "ssba":
        raise ValueError(f"We are conducting ssba backdoor attack but not {attack_name}")
    train_bd_transform = general_compose([
        (transforms.Resize(image_size), False),
        (np.array, False),
        #  'bd_trigger/ssba/record/cifar10/train_bd.hdf5'
        (SSBA_attack_replace_version(path_replace_images=f"{path_replace_image}/train_bd.hdf5"), True),
        (npFloat32ToUint8, False),
    ])
    test_bd_transform = general_compose([
        (transforms.Resize(image_size), False),
        (np.array, False),
        #  'bd_trigger/ssba/record/cifar10/test_bd.hdf5'
        (SSBA_attack_replace_version(path_replace_images=f"{path_replace_image}/test_bd.hdf5"), True),
        (npFloat32ToUint8, False),
    ])
    return train_bd_transform, test_bd_transform


def peril_img_trans_generate(attack_name, patch_mask_path, image_size, dataset):
    if attack_name != "peril":
        raise ValueError(f"We are conducting peril backdoor attack but not {attack_name}")
    trans = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        np.array,
    ])

    if dataset == "mnist":
        bd_transform = AddPatchTrigger(
            trans(Image.open(patch_mask_path).convert("L")),
        )
    else:
        bd_transform = AddPatchTrigger(
            trans(Image.open(patch_mask_path)),
        )

    train_bd_transform = general_compose([
        (transforms.Resize((image_size, image_size)), False),
        (np.array, False),
        (bd_transform, True),
        (npClipAndToUint8, False),
    ])

    test_bd_transform = general_compose([
        (transforms.Resize((image_size, image_size)), False),
        (np.array, False),
        (bd_transform, True),
        (npClipAndToUint8, False),
    ])
    return train_bd_transform, test_bd_transform


def color_img_trans_generate(attack_name, color_shift, image_size, dataset):
    if attack_name != "color":
        raise ValueError(f"We are conducting color backdoor attack but not {attack_name}")
    train_bd_transform = general_compose([
        (transforms.Resize(image_size), False),
        (np.array, False),
        (Color_backdoor_shift(color_shift), True),
        (npClipAndToUint8, False),
    ])
    test_bd_transform = general_compose([
        (transforms.Resize(image_size), False),
        (np.array, False),
        (Color_backdoor_shift(color_shift), True),
        (npClipAndToUint8, False),
    ])
    return train_bd_transform, test_bd_transform


def ftrojan_img_trans_generate(attack_name, image_size, channel_list, magnitude, YUV, window_size, pos_list):
    if attack_name != "ftrojan":
        raise ValueError(f"We are conducting ftrojan backdoor attack but not {attack_name}")
    train_bd_transform = general_compose([
        (transforms.Resize(image_size), False),
        (np.array, False),
        (ftrojann_version(YUV, channel_list, window_size, magnitude, pos_list), False),
        (npClipAndToUint8, False),
    ])
    test_bd_transform = general_compose([
        (transforms.Resize(image_size), False),
        (np.array, False),
        (ftrojann_version(YUV, channel_list, window_size, magnitude, pos_list), False),
        (npClipAndToUint8, False),
    ])
    return train_bd_transform, test_bd_transform


def bd_attack_label_trans_generate(attack_label_trans, attack_target=None, attack_label_shift_amount=None,
                                   num_classes=None):
    """
    # idea : choose which backdoor label transform you want
    :param attack_label_trans: all2all or all2one
    :param attack_target: if all2one
    :param attack_label_shift_amount: if all2all
    :param num_classes: if all2all
    :return: transform
    """

    if attack_label_trans == 'all2one':
        target_label = int(attack_target)
        bd_label_transform = AllToOne_attack(target_label)
    elif attack_label_trans == 'all2all':
        bd_label_transform = AllToAll_shiftLabelAttack(
            int(attack_label_shift_amount),
            int(num_classes)
        )
    else:
        bd_label_transform = None
    return bd_label_transform
