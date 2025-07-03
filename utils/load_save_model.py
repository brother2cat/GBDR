import torch
import os
import logging
import time

logger = logging.getLogger("mylogger")


def load_checkpoint(checkpoint_file):
    # read what the latest model file is:
    filename = checkpoint_file
    if not os.path.exists(filename):
        return None

    # load and return the checkpoint:
    else:
        # logger.info("load checkpoint")
        return torch.load(filename)


# function that saves checkpoint:
def save_checkpoint(current_epoch, checkpoint_folder, checkpoint) -> None:
    # make sure that we have a checkpoint folder:
    if not os.path.isdir(checkpoint_folder):
        try:
            os.makedirs(checkpoint_folder)
        except BaseException:
            logger.warning('| WARNING: could not create directory %s' % checkpoint_folder)
    if not os.path.isdir(checkpoint_folder):
        return False

    # write checkpoint atomically:
    try:
        file_name = f"{time.strftime('%Y%m%d%H%M%S', time.localtime())}.cpt"
        torch.save(checkpoint, checkpoint_folder + "/" + file_name)
        logger.info(f"{current_epoch} checkpoint is saved as {file_name}")
        return True
    except BaseException:
        logger.warning('| WARNING: could not write checkpoint to %s.' % checkpoint_folder)
        return False
