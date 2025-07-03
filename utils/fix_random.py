import sys
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def fix_random(
        random_seed: int = 0
) -> None:
    '''
    use to fix randomness in the script, but if you do not want to replicate experiments, then remove this can speed up your code
    :param random_seed:
    :return: None
    '''
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
