import numpy as np


class Color_backdoor_shift(object):
    def __init__(self, color_shift):
        self.color_shift = color_shift

    def __call__(self, img: np.ndarray, target: None, image_serial_id: None) -> np.ndarray:
        return img + self.color_shift
