import torch


def generate_random_mask(image, cover_rate):
    """
    cover_rate shows the Probability of 1
    """
    temp = image[:, 0, :, :]
    temp = torch.bernoulli(cover_rate * torch.ones_like(temp))
    temp = temp.unsqueeze(dim=1)
    temp = temp.expand(-1, 3, -1, -1)
    """
    temp = image[:, 0, :, :]
    temp[:, 25:32, 25:32] = 0
    temp[:, :, 0:25] = 1
    temp[:, 0:25, :] = 1
    temp = temp.unsqueeze(dim=1)
    temp = temp.expand(-1, 3, -1, -1)
    """
    return temp
