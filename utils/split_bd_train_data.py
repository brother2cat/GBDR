import torch


def split_bd_train_data(backdoor_dataset):
    clean_index = []
    poison_index = []
    for i in range(len(backdoor_dataset)):
        if backdoor_dataset[i][3] == 0:
            clean_index.append(i)
        elif backdoor_dataset[i][3] == 1:
            poison_index.append(i)
        else:
            raise ValueError("value error")
    return clean_index, poison_index
