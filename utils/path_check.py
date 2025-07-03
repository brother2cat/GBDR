import os
import sys
sys.path.append('../')
from global_param import *
os.chdir(global_path)


def path_check(path):
    """
    check the path exists, if not, create it.
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
