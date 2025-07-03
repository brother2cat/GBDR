import torch
import copy
import numpy as np
from torch.utils.data import Dataset


class dataset_wrapper_with_transform(Dataset):
    def __init__(self, obj, wrap_img_transform=None):

        # this warpper should NEVER be warp twice.
        # Since the attr name may cause trouble.
        assert not "wrap_img_transform" in obj.__dict__

        self.wrapped_dataset = obj
        self.wrap_img_transform = wrap_img_transform
        self.original_index_array = np.arange(len(self.wrapped_dataset))

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.wrapped_dataset, attr)

    def subset(self, chosen_index_list):
        self.original_index_array = self.original_index_array[chosen_index_list]

    def __getitem__(self, index):
        original_index = self.original_index_array[index]
        img, label, *other_info = self.wrapped_dataset[original_index]
        if self.wrap_img_transform is not None:
            img = self.wrap_img_transform(img)
        return img, label, *other_info

    def __len__(self):
        return len(self.original_index_array)

    def __deepcopy__(self, memo):
        return dataset_wrapper_with_transform(copy.deepcopy(self.wrapped_dataset),
                                              copy.deepcopy(self.wrap_img_transform))


class dataset_wrapper_with_augment_and_transform(Dataset):

    def __init__(self, obj, image_augment=None, wrap_img_transform=None):

        # this warpper should NEVER be warp twice.
        # Since the attr name may cause trouble.
        assert not "wrap_img_transform" in obj.__dict__

        self.wrapped_dataset = obj
        self.wrap_img_transform = wrap_img_transform
        self.image_augment = image_augment
        self.original_index_array = np.arange(len(self.wrapped_dataset))

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.wrapped_dataset, attr)

    def subset(self, chosen_index_list):
        self.original_index_array = self.original_index_array[chosen_index_list]

    def __getitem__(self, index):
        original_index = self.original_index_array[index]
        img, label, *other_info = self.wrapped_dataset[original_index]
        if self.image_augment is not None:
            img = self.image_augment(img)
        if self.wrap_img_transform is not None:
            img = self.wrap_img_transform(img)
        return img, label, *other_info

    def __len__(self):
        return len(self.original_index_array)

    def __deepcopy__(self, memo):
        return dataset_wrapper_with_augment_and_transform(copy.deepcopy(self.wrapped_dataset),
                                              copy.deepcopy(self.image_augment),
                                              copy.deepcopy(self.wrap_img_transform))