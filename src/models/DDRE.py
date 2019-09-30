# Implementation of Direct Density-Ratio Estimation (see referehces/kawahara2009.pdf)
import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count
from src.features.build_features import rolling_window


def gaussian_kernel_function(Y1, Y2, sigma):
    return np.exp(-np.linalg.norm(Y1 - Y2)**2 / (2 * sigma**2))

def get_sequences_of_samples(Y, start_idx, n, k):
    if isinstance(Y, pd.DataFrame):
        Y = Y.iloc
    return rolling_window(Y[start_idx : start_idx + n + k - 1], k)

class DensityRatioEstimation:
    def __init__(self, sigma):
        self.sigma = sigma
        self.alphas = None
        self.Y_te = None
        self.k = None
    
    def build(self, Y_re, Y_te, eps=0.001, min_delta=0.01):
        """
        param `Y_re` and `Y_te` are "rolling" windows with multidimensional time-series
        They need to has shape (n_rf, k, d)
        param `eps` is 'lerning_rate' for alphas
        param `min_delta` is minimal value of difference, regarded as improvement
        """
        assert Y_re.ndim == 3 and Y_te.ndim == 3 and Y_re.shape[1] == Y_te.shape[1]

        self.Y_te = Y_te
        self.k = Y_te.shape[1]

        n_rf = Y_re.shape[0]
        n_te = Y_te.shape[0]
        k = Y_re.shape[1]

        K = np.zeros((n_te, n_te))
        for i in range(n_te):
            for l in range(n_te):
                K[i, l] = gaussian_kernel_function(Y_te[i], Y_te[l], self.sigma)

        b = np.zeros(n_rf)
        for l in range(n_rf):
            b[l] = 1 / n_rf * np.sum([gaussian_kernel_function(Y_re[i], Y_te[l], self.sigma)
                                      for i in range(n_rf)])

        alphas = np.random.rand(n_te)

        while True:
            prev_alphas = alphas.copy()

            # Perform gradient ascent
            alphas = alphas + eps * K.T.dot((1. / K).dot(alphas))

            # Perform feasibility satisfaction
            alphas = alphas + (1 - b.dot(alphas)) * b / (b.T.dot(b))

            alphas = np.maximum(0, alphas)

            alphas = alphas / (b.T.dot(alphas))

            if np.linalg.norm(alphas - prev_alphas) < min_delta:
                break
        
        self.alphas = alphas

    def _compute_ratio_one_window(self, Y_rf):
        func = lambda pair: pair[0] * gaussian_kernel_function(Y_rf, pair[1], self.sigma)
        results = map(func, zip(self.alphas, self.Y_te))
        return sum(results)
    
    def compute_ratios(self, Y):
        data = rolling_window(Y, self.k)
        with Pool(cpu_count()) as p:
            ratios = p.map(_helper, zip([self]*data.shape[0], data))
        return ratios
    
def _helper(args):
    return args[0]._compute_ratio_one_window(args[1])


