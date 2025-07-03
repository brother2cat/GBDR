import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import ImageFilter, Image
import random
import numpy as np


class MixMatchDataset(Dataset):
    """Semi-supervised MixMatch dataset.

    Args:
        dataset: The dataset to be wrapped. bd_train_dataset_wo_transform
        transform: clean_image_transfrom
        index_list:
        with_label: True for Dx, False for Du
    """

    def __init__(self, dataset, transform, index_list, with_label):
        super(MixMatchDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.index_list = index_list
        self.with_label = with_label

        self.original_index_array = np.arange(len(self.dataset))
        self.original_index_array = self.original_index_array[self.index_list]

    def __getitem__(self, index):
        original_index = self.original_index_array[index]
        img, label, *other_information = self.dataset[original_index]
        if self.with_label:
            return self.transform(img), label, *other_information
        else:
            return self.transform(img), self.transform(img), label, *other_information

    def __len__(self):
        return len(self.original_index_array)
