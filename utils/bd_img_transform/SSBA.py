# This script is for SSBA,

from typing import Sequence
import logging
import numpy as np
import h5py
from utils.h5py_data import h5py_bdimg_label_iniimg


class SSBA_attack_replace_version(object):
    def __init__(self, path_replace_images: str) -> None:
        print('in SSBA_attack_replace_version, the real transform does not happen here, '
              'input img, target must be NONE, only image_serial_id used')

        self.replace_images = h5py_bdimg_label_iniimg(path_replace_images)

    def __call__(self, img: None,
                 target: None,
                 image_serial_id: int
                 ) -> np.ndarray:
        return self.replace_images[image_serial_id][0].transpose(1, 2, 0)
