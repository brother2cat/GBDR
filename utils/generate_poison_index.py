import torch
from typing import Callable
import numpy as np

from utils.bd_label_transform.backdoor_label_transform import AllToOne_attack, AllToAll_shiftLabelAttack


def generate_poison_index_from_label_transform(
        clean_dataset_targets: np.ndarray,
        label_transform: Callable,
        target_label: int = None,
        clean_label_attack: bool = False,
        pratio: float = None,
        train: bool = True
) -> np.ndarray:
    """
    idea:
    :param clean_dataset_targets:
    :param label_transform:
    :param target_label: if clean label attack, select from target label, if label_transform = all2one else error
    :param clean_label_attack:
    :param pratio:
    :param train: if False, pratio=1
    :return: len = len(clean_dataset_targets), 0 is clean, 1 will be poisoned
    """
    poison_index = np.zeros(len(clean_dataset_targets))
    num_samples = len(clean_dataset_targets)
    if not train:
        pratio = 1
    if isinstance(label_transform, AllToOne_attack):
        if clean_label_attack:
            if target_label is None:
                raise ValueError("clean label attack, target label should be given")
            else:
                if train:
                    index = np.random.choice(np.where(clean_dataset_targets == target_label)[0],
                                             int(pratio * num_samples),
                                             replace=False)
                    poison_index[index] = 1
                    return poison_index
                else:
                    index = np.random.choice(np.arange(num_samples),
                                             int(pratio * num_samples),
                                             replace=False)
                    poison_index[index] = 1
                    return poison_index
    index = np.random.choice(np.arange(num_samples), int(pratio * num_samples), replace=False)
    poison_index[index] = 1
    return poison_index
