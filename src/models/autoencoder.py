import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_size, output_size, lambda_fraction, dropout_fraction):
        super().__init__(self)
        self.input_dropout = nn.Dropout(lambda_fraction)
        self.encoder = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout_fraction)
        self.decoder = nn.Linear(output_size, input_size)

        self.last_activations = None

    def forward(self, X):
        inter = self.encoder(X)
        inter = F.sigmoid(inter)
        self.last_activations = inter
        inter = self.dropout(inter)
        inter = self.decoder(inter)
        return F.sigmoid(inter)

class CustomLoss(nn.Module):
    def __init__(self, model, alpha, beta, sparsity):
        super().__init__(self)
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.sparsity = sparsity
        self.mse = nn.MSELoss

    def forward(self, input, target):
        mse = 1/2 * self.mse(input, target)
        weight_regularization = self.alpha/2 * (self.model.encoder.weight.norm() +\
            self.model.decoder.weight.norm())

        # TODO: misunderstanding sparsity
        sparsity_loss = torch.tensor([0.])
        for value in self.model.last_activations.mean(axis=0):
            sparsity_loss += self.beta * F.kl_div(value, self.sparsity)

        return mse + weight_regularization + sparsity_loss

# TODO: check this
class ConvFeatrueMapping(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling_size):
        super().__init__(self)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.pooling_size = pooling_size

    def forward(self, X):
        inter = F.sigmoid(self.conv(X))
        return F.max_pool1d(inter, kernel_size=self.pooling_size)