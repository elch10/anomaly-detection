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
    return rolling_window(Y[start_idx : start_idx + n + k], k)

class DensityRatioEstimation:
    def __init__(self, sigma):
        self.sigma = sigma
        self.Y_rf = None
        self.Y_te = None
        self.k = None
        self.alphas = None
        self.b = None
    
    def _feasibility(self):
        """
        Performs feasibility satisfaction
        """
        self.alphas = self.alphas + (1 - self.b.T.dot(self.alphas)) * self.b / (self.b.T.dot(self.b))
        self.alphas = np.maximum(0, self.alphas)
        self.alphas = self.alphas / (self.b.T.dot(self.alphas))

    def _compute_b(self):
        n_rf = self.Y_rf.shape[0]
        b = np.zeros(n_rf)
        for l in range(n_rf):
            b[l] = np.sum([gaussian_kernel_function(self.Y_rf[i], self.Y_te[l], self.sigma)
                           for i in range(n_rf)]) / n_rf
        self.b = b


    def build(self, Y_rf, Y_te, eps=0.001, min_delta=0.01, iterations=20):
        """
        param `Y_rf` and `Y_te` are "rolling" windows with multidimensional time-series
        They need to has shape (n_rf, k, d)
        `n_rf` need to be equal to `n_te`
        param `eps` is 'lerning_rate' for alphas
        param `min_delta` is minimal value of difference, regarded as improvement
        param `iterations` is maximum amount of iterations used to find proper alphas
        """
        assert Y_rf.shape == Y_te.shape

        self.Y_rf = Y_rf
        self.Y_te = Y_te
        self.k = Y_te.shape[1]

        n_te = Y_te.shape[0]

        K = np.zeros((n_te, n_te))
        for i in range(n_te):
            for l in range(n_te):
                K[i, l] = gaussian_kernel_function(Y_te[i], Y_te[l], self.sigma)

        self._compute_b()
        self.alphas = np.random.rand(n_te)

        for _ in range(iterations):
            prev_alphas = self.alphas.copy()

            # Perform gradient ascent
            self.alphas = self.alphas + eps * K.T.dot((1. / K).dot(self.alphas))

            self._feasibility()

            if np.linalg.norm(self.alphas - prev_alphas) < min_delta:
                break
        

    def compute_ratio_one_window(self, Y):
        func = lambda pair: pair[0] * gaussian_kernel_function(Y, pair[1], self.sigma)
        results = list(map(func, zip(self.alphas, self.Y_te)))
        return sum(results)

    def compute_likelihood_ratio(self):
        return np.sum(np.log(self.compute_ratio_windows(self.Y_te)))
        
    def compute_ratio_windows(self, Y):
        """
        Y need to has shape (n_rf, k, d)
        """
        assert Y.shape[1] == self.k and Y.ndim == 3
        
        ratios = np.zeros(Y.shape[0])

        for i, sample in enumerate(Y):
            ratios[i] = self.compute_ratio_one_window(sample)

        # with Pool(cpu_count()) as p:
        #     ratios = p.map(_helper, zip([self]*Y.shape[0], Y))
        
        # In original paper we just return computed ratios
        # But it is seems as error, because it's somewhat confusing
        return 1. / np.array(ratios)
        # return np.array(ratios)
    
    def compute_ratios_df(self, df):
        """
        df need to be the shape (len, d)
        """
        return self.compute_ratio_windows(rolling_window(df, self.k))

    def update_by_new_sample(self, y, lerning_rate, reg_parameter):
        """
        `y` is the new rolling window from the time (n_te + 1) to (n_te + 1 + k)
        param `y` must have shape (k, d)
        param `lerning_rate` is the learning rate that controls the adaptation sensitivity to the new sample
        param `reg_parameter` is the regularization parameter
        """

        new_alphas = np.zeros_like(self.alphas)
        new_alphas[:-1] = (1 - lerning_rate * reg_parameter) * self.alphas[1:]
        new_alphas[-1] = lerning_rate / self.compute_ratio_windows(y[None, ...])
        self.alphas = new_alphas

        self.Y_rf[:-1] = self.Y_rf[1:]
        self.Y_rf[-1] = self.Y_te[0]
        
        self.Y_te[:-1] = self.Y_te[1:]
        self.Y_te[-1] = y

        self._compute_b()
        self._feasibility()

        
    
def _helper(args):
    return args[0].compute_ratio_one_window(args[1])


def kernel_width_selection(Y_rf, Y_te, candidates, R=3):
    """
    Selects optimal gaussian width.
    param `Y_rf` and `Y_te` are "rolling" windows
    They need to has shape (n_rf, k, d)
    """
    Y_te_copy = np.copy(Y_te)
    np.random.shuffle(Y_te_copy)
    Y_te_arrays = np.split(Y_te_copy, R)

    J = np.zeros_like(candidates, dtype=float)

    for i, width in enumerate(candidates):
        dre = DensityRatioEstimation(width)
        J_r = np.zeros(R)
        for r in range(R):
            Y_te_split = Y_te_arrays[:r] + Y_te_arrays[r+1:]
            dre.build(Y_rf, np.vstack(Y_te_split))
            ratios = dre.compute_ratio_windows(Y_te_arrays[r])
            J_r[r] = np.mean(np.log(ratios))
        J[i] = np.mean(J_r)
    
    optimal_idx = np.argmax(J)
    return candidates[optimal_idx]


