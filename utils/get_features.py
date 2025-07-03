import torch
import numpy as np


def get_features(model, target_layer, data_loader, device):
    """
    get the features of dataloader from model
    if True, dataloader: img, label, original_index, poison_or_not, original_target, else: img, label
    :param model:
    :param target_layer:
    :param data_loader:
    :param device:
    :return:
    """

    def feature_hook(module, input, output):
        global feature_vector
        # access the layer output and convert it to a feature vector
        feature_vector = output
        feature_vector = torch.flatten(feature_vector, 1)

    h = target_layer.register_forward_hook(feature_hook)

    model.eval()
    # collect feature vectors
    features = []
    labels = []
    # poi_indicator = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, *other_info) in enumerate(data_loader):
            global feature_vector
            # Fetch features
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # move all tensor to cpu to save memory
            current_feature = feature_vector.detach().cpu().numpy()
            if len(other_info) == 0:
                current_labels = targets.cpu().numpy()
            else:
                current_labels = [999 if other_info[1][i] else targets[i].cpu().item() for i in range(len(targets))]
                current_labels = np.array(current_labels)

            features.append(current_feature)
            labels.append(current_labels)

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    h.remove()  # Rmove the hook

    return features, labels

