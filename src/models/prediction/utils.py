import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class RunningLoss:
    def __init__(self):
        self.losses = []

    def update(self, loss, size):
        self.losses += [loss] * size

    def avg(self):
        return np.mean(self.losses)

def to_dataloader(X, y, params):
    X = torch.tensor(X).float()
    y = torch.tensor(y).squeeze().float()
    
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, **params)
