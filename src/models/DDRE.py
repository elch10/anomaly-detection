# Implementation of Direct Density-Ratio Estimation (see referehces/kawahara2009.pdf)
import math
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import numba as nb
from sklearn.preprocessing import StandardScaler

from .utils import gaussian_kernel_function
from src.utils import inverse_ids
from src.features.build_features import rolling_window


cpu_cnt = cpu_count()

_spec = [
    ('sigma', nb.float64),
    ('Y', nb.float64[:, :]),
    ('k', nb.int32),
    ('window_width', nb.int64),
    ('last_procceed', nb.int64),
    ('n_rf_te', nb.int32),
    ('alphas', nb.float64[:]),
    ('b', nb.float64[:])
]

@nb.jitclass(_spec)
class DensityRatioEstimation:
    def __init__(self, sigma, window_width, n_rf_te):
        self.sigma = sigma
        self.Y = np.array([[0.]], dtype=np.float64)
        self.window_width = window_width
        self.last_procceed = 0
        self.n_rf_te = n_rf_te
        self.alphas = np.array([0.], dtype=np.float64)
        self.b = np.array([0.], dtype=np.float64)

    def get_rf(self, shift):
        return self.Y[self.last_procceed + shift : self.last_procceed + self.window_width + shift]

    def get_te(self, shift):
        return self.get_rf(self.n_rf_te + self.window_width - 1 + shift)
    
    def _feasibility(self):
        """
        Performs feasibility satisfaction
        """
        a = (1 - np.dot(self.b.T, self.alphas)) * self.b
        a /= (np.dot(self.b.T, self.b))
        self.alphas += a
        self.alphas = np.maximum(0, self.alphas)
        self.alphas = self.alphas / np.dot(self.b.T, self.alphas)

    def _compute_b(self):
        b = np.zeros(self.n_rf_te, dtype=np.float64)
        for l in range(self.n_rf_te):
            s = 0.
            for i in range(self.n_rf_te):
                s += gaussian_kernel_function(self.get_rf(i), self.get_te(l), self.sigma)
            b[l] = s / self.n_rf_te
        self.b = b

    def build(self, Y, eps=0.001, min_delta=0.01, iterations=100):
        """
        param `Y` is a multidimensional time-series with shape (n, d)
        param `eps` is 'lerning_rate' for alphas
        param `min_delta` is minimal value of difference, regarded as improvement
        param `iterations` is maximum amount of iterations used to find proper alphas
        """
        self.Y = Y
        n_rf_te = np.int64(self.n_rf_te)

        K = np.zeros((n_rf_te, n_rf_te), dtype=np.float64)
        for i in range(n_rf_te):
            for l in range(n_rf_te):
                K[i, l] = gaussian_kernel_function(self.get_te(i), self.get_te(l), self.sigma)

        self._compute_b()
        self.alphas = np.random.rand(n_rf_te).astype(np.float64)

        for _ in range(iterations):
            prev_alphas = self.alphas.copy()

            # Perform gradient ascent
            temp = eps * np.dot(K.T, np.dot((1. / K).astype(np.float64), self.alphas))
            self.alphas += temp

            self._feasibility()

            if np.linalg.norm(self.alphas - prev_alphas) < min_delta:
                break
        
    def compute_ratio_one_window(self, Y):
        res = np.zeros_like(self.alphas)
        for i in range(self.alphas.shape[0]):
            res[i] = gaussian_kernel_function(Y, self.get_te(i), self.sigma)
        res *= self.alphas
        return np.sum(res)

    def compute_likelihood_ratio(self):
        ratios = np.zeros(self.n_rf_te, dtype=np.float64)
        for i in range(self.n_rf_te):
            ratios[i] = self.compute_ratio_one_window(self.get_te(i))
        return np.sum(np.log(ratios))
        
    def compute_ratio_windows(self, Y):
        """
        Y need to has shape (n + window_width, d)
        """
        ratios = np.zeros(Y.shape[0] - self.window_width, dtype=np.float64)

        for i in range(self.window_width, Y.shape[0]):
            ratios[i-self.window_width] = self.compute_ratio_one_window(Y[i-self.window_width:i])
        
        return ratios

    def update_by_next_sample(self, learning_rate, reg_parameter):
        """
        Updates model by using next sample from data
        param `learning_rate` is the learning rate that controls the adaptation sensitivity to the new sample
        param `reg_parameter` is the regularization parameter
        """
        if self.last_procceed + 2 * self.n_rf_te + 2 * self.window_width - 1 >= self.Y.shape[0]:
            print("Cannot update by next sample. Last sample already was used")
            return

        self.last_procceed += 1

        new_alphas = np.zeros_like(self.alphas)
        new_alphas[:-1] = (1 - learning_rate * reg_parameter) * self.alphas[1:]
        new_alphas[-1] = learning_rate / self.compute_ratio_one_window(self.get_te(self.n_rf_te))
        self.alphas = new_alphas

        self._compute_b()
        self._feasibility()

@nb.njit(parallel=True)
def kernel_sigma_selection(df, window_width, candidates, R=4):
    """
    Selects optimal gaussian width.
    param `df` is a "rolling" window. It musts have shape of (n, d), n>=window_width
    param `R` characterize the number of each split chunks. The `n` must be divisible by `2 * R - 1`
    The first chunk would be used as reference sample. And others `R-1` as test in cross-validation
    """
    if len(candidates) == 1:
        return np.zeros(0, dtype=np.float64), candidates[0]

    n = df.shape[0] - 2 * window_width
    assert n % (2 * R - 1) == 0

    chunk_size = n // (2 * R - 1)
    
    Y_rf = df[:(R - 1) * chunk_size + window_width]
    Y_te_arrays = df[(R - 1) * chunk_size + window_width:]

    J = np.zeros(len(candidates), dtype=np.float64)

    for i in nb.prange(len(candidates)):
        dre = DensityRatioEstimation(candidates[i], window_width, chunk_size * (R - 1))
        J_r = np.zeros(R)
        for r in range(R):
            Y_ = np.concatenate((Y_rf, np.vstack((Y_te_arrays[:r*chunk_size+window_width], Y_te_arrays[(r+1)*chunk_size+window_width:]))), axis=0)
            dre.build(Y_)
            ratios = dre.compute_ratio_windows(Y_te_arrays[r*chunk_size: (r+1)*chunk_size+window_width])
            J_r[r] = np.mean(np.log(ratios))
        J[i] = np.mean(J_r)
    
    J[np.isnan(J)] = -2e9
    optimal_idx = np.argmax(J)
    return J, candidates[optimal_idx]

@nb.njit(parallel=True)
def ddre_ratios(df,
                window_width,
                sigma_candidates,
                chunk_size,
                R,
                n_rf_te,
                eps=0.001, min_delta=0.01, iterations=100,
                learning_rate=1, reg_parameter=0.01,
                tresh=-1,
                verbose=False):
    """
    Computes ratios over all `df` with shape (n, d). It need to be the numpy array, not pd.DataFrame!
    param `window_width` is a width of rolling window
    param `sigma_candidates` and `R` used in `kernel_sigma_selection`. Refer there for documentation
    param `chunk_size` is the size of chunk used in cross-validation
    param `n_rf_te` characterizes size of reference and test samples. They are equal due matrix multiplication
    by transposed itself (number of rows and columns must be equal)
    param `build_args` and `update_args` used in model building and parameter updating
    param `tresh` is treshold from original paper (if it is -1, then it is ignored)
    param `verbose` characterize whether to print progress every 5%
    
    Returns:
    `ratios` - is a computed ratios of probability densities
    `change_points` - indexes of changing. This is not empty if you specified right `tresh`
    """
    # print('Finding optimal sigma...')

    data = df[:chunk_size * (2 * R - 1) + 2 * window_width]
    
    J, optimal_sigma = kernel_sigma_selection(data, window_width, sigma_candidates, R)
    print('Optimal sigma is:', optimal_sigma)

    n = df.shape[0]

    ratios = np.zeros(n, dtype=np.float64)
    change_points = []

    five_percent_size = math.ceil(n / 20)

    piece_size = n // cpu_cnt
    pieces_cnt = (n + piece_size - 1) // piece_size
    
    if piece_size <= n_rf_te * 2 + window_width * 2:
        piece_size = n
        pieces_cnt = 1
    
    print('Computing ratios in parrallel with', pieces_cnt, 'threads')

    for i in nb.prange(pieces_cnt):
        start_left = i*piece_size+n_rf_te*2+window_width*2
        t = start_left

        if i == pieces_cnt - 1:
            right = n - 1
        else:
            right = (i+1)*piece_size

        while t + 1 < right:
            dre = DensityRatioEstimation(optimal_sigma, window_width, n_rf_te)
            dre.build(df[i*piece_size:right+1], eps, min_delta, iterations)

            while t + 1 < right:
                if verbose and (t % five_percent_size == 0):
                    print(i, 5 * t // five_percent_size, '%')
                
                # numba can't compile this
                # dre.update_by_next_sample(Y[t], **update_args)
                # so hardcode this
                dre.update_by_next_sample(learning_rate, reg_parameter)

                ratios[t] = dre.compute_likelihood_ratio()
                t += 1
                if tresh != -1 and ratios[t - 1] > tresh:
                    change_points.append(t - 1)
                    t += n_rf_te + n_rf_te - 1
                    break

        ratios[i*piece_size: start_left] = ratios[start_left+1]

    return ratios, change_points

def kernel_width_selection(Y, width_candidates, other_params):
    """
    Finds appropriate width of rolling window
    param `Y` - data with shape (n, d)
    param `width_candidates` - probable candidates for choosing width
    param `other_params` - params that would be passed to `ddre_ratios_df` function

    Returns tuple of elements:
    1. Sum squared distance beetwen mean of non change-points derivatives of ratios and
    change-points derivatives of ratios for every width candidate in `width_candidates`
    2. Optimal width that corresponds to maximal sum squared distance
    """
    ssds = []
    if len(width_candidates) == 1:
        return ssds, width_candidates[0]
    
    for candidate in width_candidates:
        other_params['window_width'] = candidate
        print('Candidate', candidate)
        ratios, _ = ddre_ratios(Y, **other_params)

        abnormal_idxs, _ = find_peaks(ratios, distance=candidate)
        normal_idxs = inverse_ids(abnormal_idxs, ratios.shape[0])

        scaled = StandardScaler().fit_transform(ratios[ratios.nonzero()[0], None]).ravel()
        ratios[ratios.nonzero()[0]] = scaled

        abnormal = ratios[abnormal_idxs]

        ssd = np.nanmean((abnormal[~np.isnan(abnormal)] - np.nanmean(ratios[normal_idxs])) ** 2)
        ssds.append(ssd)
    return ssds, width_candidates[np.nanargmax(ssds)]


def compute_ratios(y, width_candidates, params, ratio=0.3):
    """
    Finds density ratios with hyperparameter search
    """
    print('Finding hyperparams...')
    _, optimal = kernel_width_selection(y[:int(y.shape[0]*ratio)], width_candidates, params)
    params['window_width'] = int(optimal)
    print(f'\nOptimal width is {optimal}\n')
    
    print('Starting compute ratios...')
    ratios, chng_pts = ddre_ratios(y, **params)
    peaks, _ = find_peaks(ratios, distance=optimal)
    
    return ratios, chng_pts, peaks
