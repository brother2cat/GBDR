
import numpy as np
import torch
from typing import Optional, Union
from torchvision.transforms import Resize, ToTensor, ToPILImage


class AddPatchTrigger(object):
    '''
    assume init use HWC format
    '''

    def __init__(self, trigger_array: np.ndarray):
        self.trigger_array = trigger_array

    def __call__(self, img, target=None, image_serial_id=None):
        return self.add_trigger(img)

    def add_trigger(self, img: np.ndarray):
        return np.clip(img + self.trigger_array, 0, 255)


class AddMaskPatchTrigger(object):
    def __init__(self,
                 trigger_array: Union[np.ndarray, torch.Tensor],
                 ):
        self.trigger_array = trigger_array

    def __call__(self, img, target=None, image_serial_id=None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return img * (self.trigger_array == 0) + self.trigger_array * (self.trigger_array > 0)


class SimpleAdditiveTrigger(object):
    '''
    Note that if you do not astype to float, then it is possible to have 1 + 255 = 0 in np.uint8 !
    '''

    def __init__(self,
                 trigger_array: np.ndarray,
                 ):
        self.trigger_array = trigger_array.astype(np.float)

    def __call__(self, img, target=None, image_serial_id=None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return np.clip(img.astype(np.float) + self.trigger_array, 0, 255).astype(np.uint8)



