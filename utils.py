import random

import numpy as np
import torch
from torch.utils.data import random_split


def train_test_split_dataset(dataset, train_test_ratio):
    train_size = int(len(dataset) * train_test_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


def train_test_split(data, train_test_ratio, shuffle=False):
    if shuffle:
        data = random.sample(data, len(data))
    split_index = int(len(data) * train_test_ratio)
    train_data = data[:split_index]
    test_data = data[split_index]
    return train_data, test_data


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
