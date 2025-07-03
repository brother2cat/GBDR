import numpy as np


def sub_sample_index(label_all, ratio=None, selected_classes=None, max_num_samples=None):
    # subsample the data with ratio for each classes
    class_unique = np.unique(label_all)
    if selected_classes is not None:
        # find the intersection of selected_classes and class_unique
        class_unique = np.intersect1d(
            class_unique, selected_classes, assume_unique=True, return_indices=False)
    select_idx = []
    if max_num_samples is not None:
        print('max_num_samples is given, use sample number limit now.')
        total_selected_samples = np.sum(
            [np.where(label_all == c_idx)[0].shape[0] for c_idx in class_unique])
        ratio = np.min([total_selected_samples, max_num_samples]
                       ) / total_selected_samples

    for c_idx in class_unique:
        sub_idx = np.where(label_all == c_idx)
        sub_idx = np.random.choice(sub_idx[0], int(
            ratio * sub_idx[0].shape[0]), replace=False)
        select_idx.append(sub_idx)
    sub_idx = np.concatenate(select_idx, -1).reshape(-1)
    # shuffle the sub_idx
    sub_idx = sub_idx[np.random.permutation(sub_idx.shape[0])]
    return sub_idx
