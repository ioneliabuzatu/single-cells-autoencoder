import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def correlation_score_fn(y_true, y_pred):
    """Scores the predictions according to the competition rules. 
    It is assumed that the predictions are not constant.
    Returns the average of each sample's Pearson correlation coefficient"""
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values

    assert y_true.shape == y_pred.shape, print(f"{y_true.shape} vs {y_pred.shape}.")

    corr_sum = 0
    for i in range(len(y_true)):
        corr_sum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corr_sum / len(y_true)


class MyDataset(Dataset):

    def __init__(self, inputs, train: bool, whoami: str):
        if type(inputs) == pd.DataFrame: inputs = inputs.values

        if whoami == 'ionelia':
            if train:
                x = inputs[:50]
            else:
                x = inputs[50:]
        else:
            if train:
                x = inputs[:50_000]
            else:
                x = inputs[50_000:]

        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

class DatasetWithY(Dataset):

    def __init__(self, inputs, targets, train: bool, whoami: str):
        if type(inputs) == pd.DataFrame: inputs = inputs.values
        if type(targets) == pd.DataFrame: targets = targets.values

        if whoami == 'ionelia':
            if train:
                x = inputs[:50]
                y = targets[:50]
            else:
                x = inputs[50:]
                y = targets[50:]
        else:
            if train:
                x = inputs[:50_000]
                y = targets[:50_000]
            else:
                x = inputs[50_000:]
                y = targets[50_000:]

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

class TestDataset(Dataset):

    def __init__(self, inputs):

        if type(inputs) == pd.DataFrame:
            inputs = inputs.values

        self.x = torch.tensor(inputs, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]