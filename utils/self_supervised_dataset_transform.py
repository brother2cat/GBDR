import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import ImageFilter, Image
import random
import numpy as np

from torchvision.transforms.functional import InterpolationMode


class GaussianBlur(object):

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))

        return x


def get_transform_self(input_height, input_width):
    # idea : given name, return the final implememnt transforms for the dataset during self-supervised learning
    transforms_list = [transforms.Resize((input_height, input_width)),
                       transforms.RandomResizedCrop(size=(input_height, input_width), scale=(0.2, 1.0),
                                                    ratio=(0.75, 1.3333),
                                                    interpolation=InterpolationMode.BICUBIC),
                       transforms.RandomHorizontalFlip(p=0.5),
                       transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=(0.6, 1.4),
                                                                                          contrast=(0.6, 1.4),
                                                                                          saturation=(0.6, 1.4),
                                                                                          hue=(-0.1, 0.1))]),
                                              p=0.8), transforms.RandomGrayscale(p=0.2),
                       transforms.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5)]

    return transforms.Compose(transforms_list)


class SelfSupervisedDataset(Dataset):
    def __init__(self, dataset, slsv_transform, img_transform=None):

        # this warpper should NEVER be warp twice.
        # Since the attr name may cause trouble.
        assert not "img_transform" in dataset.__dict__

        self.dataset = dataset
        self.slsv_transform = slsv_transform
        self.img_transform = img_transform
        self.original_index_array = np.arange(len(self.dataset))

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.dataset, attr)

    def subset(self, chosen_index_list):
        self.original_index_array = self.original_index_array[chosen_index_list]

    def __getitem__(self, index):
        original_index = self.original_index_array[index]
        img, *other_info = self.dataset[original_index]
        img1 = self.slsv_transform(img)
        img2 = self.slsv_transform(img)
        if self.img_transform is not None:
            img1 = self.img_transform(img1)
            img2 = self.img_transform(img2)
        return img1, img2

    def __len__(self):
        return len(self.original_index_array)
