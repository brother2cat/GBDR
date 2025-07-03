import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def compute_loss_per_example(model, data, device):
    """
    Calculate the loss of per example (CrossEntropy loss)
    :param: model
    :param: data, dataset or dataloader
    :return: list
    """
    if isinstance(data, Dataset):
        data = DataLoader(data, batch_size=128, shuffle=False)
    elif isinstance(data, DataLoader):
        pass
    else:
        raise TypeError(f"data's type is {type(data)}, it should be DataLoader or Dataset")

    criterion = nn.CrossEntropyLoss(reduction='none')
    model.eval()

    loss_dict = {}
    current_example = 0  # counter
    model = model.to(device)
    with torch.no_grad():
        for batch_id, (x, label, *other_info) in enumerate(data):
            x, label = x.to(device), label.to(device)
            num = x.shape[0]
            logit = model(x)
            loss = criterion(logit, label)

            if len(other_info) == 0:
                # No extra info, use the original index
                index_list = range(current_example, current_example + num)
                loss_list = loss.tolist()
                temp = dict(zip(index_list, loss_list))
            else:
                original_index = other_info[0].tolist()
                loss_list = loss.tolist()
                temp = dict(zip(original_index, loss_list))
            loss_dict.update(temp)
    return loss_dict


def compute_loss_per_example_current_dataset_index(model, data, device):
    """
    Calculate the loss of per example (CrossEntropy loss)
    :param: model
    :param: data, dataset or dataloader
    :return: list
    """
    if isinstance(data, Dataset):
        data = DataLoader(data, batch_size=128, shuffle=False)
    elif isinstance(data, DataLoader):
        pass
    else:
        raise TypeError(f"data's type is {type(data)}, it should be DataLoader or Dataset")

    criterion = nn.CrossEntropyLoss(reduction='none')
    model.eval()

    loss_dict = {}
    current_example = 0  # counter
    model = model.to(device)
    with torch.no_grad():
        for batch_id, (x, label, *other_info) in enumerate(data):
            x, label = x.to(device), label.to(device)
            num = x.shape[0]
            logit = model(x)
            loss = criterion(logit, label)

            index_list = range(current_example, current_example + num)
            loss_list = loss.tolist()
            temp = dict(zip(index_list, loss_list))

            loss_dict.update(temp)

            current_example = current_example + num
    return loss_dict
