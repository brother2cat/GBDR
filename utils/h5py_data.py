import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage
import h5py


class h5py_bdimg_label_iniimg(Dataset):
    def __init__(self, file_name) -> None:
        super().__init__()
        self.file_name = file_name

    def __getitem__(self, index):
        with h5py.File(self.file_name, 'r') as f:
            return f['fingerprinted_images'][index], int(f['label'][index]), f['initial_images'][index]

    def __len__(self):
        with h5py.File(self.file_name, 'r') as f:
            return len(f['image'])


class h5py_image_index_loss_per_sample(Dataset):
    """
    original_index
    poison_or_not
    loss_per_sample
    original_label
    poisoned_label
    """

    def __init__(self, file_name) -> None:
        super().__init__()
        self.file_name = file_name

    def __getitem__(self, index):
        with h5py.File(self.file_name, 'r') as f:
            return f['original_index'][index], \
                f['poison_or_not'][index], \
                f['loss_per_sample'][index], \
                f['original_label'][index], \
                f['poisoned_label'][index]

    def __len__(self):
        with h5py.File(self.file_name, 'r') as f:
            return len(f['original_index'])


class h5py_image_label(Dataset):

    def __init__(self, file_name) -> None:
        super().__init__()
        self.file_name = file_name

    def __getitem__(self, index):
        with h5py.File(self.file_name, 'r') as f:
            return f['image'][index], f['label'][index]

    def __len__(self):
        with h5py.File(self.file_name, 'r') as f:
            return len(f['image'])


class h5py_recon_image_other_info(Dataset):

    def __init__(self, file_name) -> None:
        super().__init__()
        self.file_name = file_name
        self.to_pil = ToPILImage()

    def __getitem__(self, index):
        with h5py.File(self.file_name, 'r') as f:
            return self.to_pil(torch.from_numpy(f['image'][index])), \
                f['info0'][index], \
                f['info1'][index], \
                f['info2'][index], \
                f['info3'][index]

    def __len__(self):
        with h5py.File(self.file_name, 'r') as f:
            return len(f['image'])


class h5py_recon_relabel_image_other_info(Dataset):

    def __init__(self, file_name) -> None:
        super().__init__()
        self.file_name = file_name

    def __getitem__(self, index):
        with h5py.File(self.file_name, 'r') as f:
            return torch.from_numpy(f['image'][index]), \
                f['relabel'][index], \
                f['info0'][index], \
                f['info1'][index], \
                f['info2'][index]

    def __len__(self):
        with h5py.File(self.file_name, 'r') as f:
            return len(f['image'])
