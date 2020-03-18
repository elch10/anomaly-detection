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

def get_prev_states(model, batch_size):
    outs = model.outs
    if model.cell_states is None and not outs:
        return None
    keys = sorted(outs.keys())
    hn = []
    for key in keys:
        if 'lstm' in key:
            hn.append(outs[key].detach().numpy())

    cn = torch.tensor(model.cell_states)[:, :batch_size]
    hn = torch.tensor(np.array(hn))[:, :batch_size]
    
    return (hn, cn)

def to_dataloader(X, y, params):
    X = torch.tensor(X).float()
    y = torch.tensor(y).squeeze().float()
    
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, **params)
