'''This script is to generate a black image with repetitive white square at the whole image'''
import numpy as np
import argparse
from PIL import Image


def generate_white_square_image(image_size, alpha, gap, width):
    """
    :param image_size:
    :param alpha: alpha is the intensity of the bright pixels in the trigger
                  and intensity of the rest of the pixel is 0
    :param gap: gap is the distance between two adjacent set of bright pixels
    :param width: it is the width of each set of bright pixels.
    :return:
    """
    black_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    up_left_location = np.arange(0, image_size, gap)
    for up_left_row in up_left_location:
        for up_left_col in up_left_location:
            up_right = (up_left_row, (up_left_col + width) if (up_left_col + width) < image_size else image_size)
            down_left = ((up_left_row + width) if (up_left_row + width) < image_size else image_size, up_left_col)
            down_right = ((up_left_row + width) if (up_left_row + width) < image_size else image_size,
                          (up_left_col + width) if (up_left_col + width) < image_size else image_size)
            black_image[up_left_row: (up_left_row + width) if (up_left_row + width) < image_size else image_size,
                        up_left_col: (up_left_col + width) if (up_left_col + width) < image_size else image_size,
                        :] = 40
    return black_image


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--image_size', type=int, default=64)
    args.add_argument('--alpha', type=int, default=40)
    args.add_argument('--gap', type=int, default=2)
    args.add_argument('--width', type=int, default=1)
    args.add_argument('--output_path', type=str, default='./trigger_image_mini .png')
    args = args.parse_args()
    image = generate_white_square_image(
        args.image_size,
        args.alpha,
        args.gap,
        args.width,
    )
    Image.fromarray(image).save(args.output_path)
