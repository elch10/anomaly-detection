# Implementation of Direct Density-Ratio Estimation (see referehces/kawahara2009.pdf)
import numpy as np
import pandas as pd
from numba import jitclass, int32, float64, types
from numba.typed import List
import math

from .utils import gaussian_kernel_function
from src.utils import inverse_ids
from src.features.build_features import rolling_window

from scipy.signal import find_peaks

_spec = [
    ('sigma', float64),
    ('Y_rf', types.ListType(float64[::, ::1])),
    ('Y_te', types.ListType(float64[::, ::1])),
    ('k', int32),
    ('alphas', float64[:]),
    ('b', float64[:])
]

@jitclass(_spec)
class DensityRatioEstimation:
    def __init__(self, sigma):
        self.sigma = sigma
        l = List()
        l.append(np.array([[0.]], dtype=np.float64))
        self.Y_rf = l

        l = List()
        l.append(np.array([[0.]], dtype=np.float64))
        self.Y_te = l
        
        self.k = 0
        self.alphas = np.array([0.], dtype=np.float64)
        self.b = np.array([0.], dtype=np.float64)
    
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
        n_rf = len(self.Y_rf)
        b = np.zeros(n_rf, dtype=np.float64)
        for l in range(n_rf):
            s = 0.
            for i in range(n_rf):
                s += gaussian_kernel_function(self.Y_rf[i], self.Y_te[l], self.sigma)
            b[l] = s / n_rf
        self.b = b


    def build(self, Y_rf, Y_te, eps=0.001, min_delta=0.01, iterations=100):
        """
        param `Y_rf` and `Y_te` are "rolling" windows with multidimensional time-series
        They need to has shape (n_rf, k, d)
        `n_rf` need to be equal to `n_te`
        param `eps` is 'lerning_rate' for alphas
        param `min_delta` is minimal value of difference, regarded as improvement
        param `iterations` is maximum amount of iterations used to find proper alphas
        """
        assert len(Y_rf) == len(Y_te)

        self.Y_rf = Y_rf
        self.Y_te = Y_te
        self.k = Y_te[0].shape[0]

        n_te = len(Y_te)

        K = np.zeros((n_te, n_te), dtype=np.float64)
        for i in range(n_te):
            for l in range(n_te):
                K[i, l] = gaussian_kernel_function(Y_te[i], Y_te[l], self.sigma)

        self._compute_b()
        self.alphas = np.random.rand(n_te).astype(np.float64)

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
            res[i] = gaussian_kernel_function(Y, self.Y_te[i], self.sigma)
        res *= self.alphas
        return np.sum(res)

    def compute_likelihood_ratio(self):
        rts = self.compute_ratio_windows(self.Y_te)
        log = np.log(rts)
        s = np.sum(log)
        return s
        
    def compute_ratio_windows(self, Y):
        """
        Y need to has shape (n_rf, k, d)
        """
        assert Y[0].shape[0] == self.k and Y[0].ndim == 2
        
        ratios = np.zeros(len(Y), dtype=np.float64)

        for i in range(len(Y)):
            ratios[i] = self.compute_ratio_one_window(Y[i])

        return ratios

    def update_by_new_sample(self, y, lerning_rate, reg_parameter):
        """
        `y` is the new rolling window from the time (n_te + 1) to (n_te + 1 + k)
        param `y` must have shape (k, d)
        param `lerning_rate` is the learning rate that controls the adaptation sensitivity to the new sample
        param `reg_parameter` is the regularization parameter
        """

        new_alphas = np.zeros_like(self.alphas)
        new_alphas[:-1] = (1 - lerning_rate * reg_parameter) * self.alphas[1:]
        new_alphas[-1] = lerning_rate / self.compute_ratio_one_window(y)
        self.alphas = new_alphas

        self.Y_rf.pop(0)
        self.Y_rf.append(self.Y_te[0])
        
        self.Y_te.pop(0)
        self.Y_te.append(y)

        self._compute_b()
        self._feasibility()

# def get_rolling_window(Y, start, )

def kernel_sigma_selection(Y, candidates, R=4):
    """
    Selects optimal gaussian width.
    param `Y` is a "rolling" window. It musts have shape of (n, k, d)
    param `R` characterize the size of each split chunk. The `n` must be divisible by `2 * R - 1`
    The first chunk would be used as reference sample. And others `R-1` as test in cross-validation
    """
    n = len(Y)
    assert n % (2 * R - 1) == 0

    chunk_size = n // (2 * R - 1)
    
    Y_rf = Y[:(R - 1) * chunk_size]
    Y_te_copy = np.copy(Y[(R - 1) * chunk_size:])
    np.random.shuffle(Y_te_copy)
    Y_te_arrays = np.split(Y_te_copy, R)

    J = np.zeros_like(candidates, dtype=np.float64)

    for i, width in enumerate(candidates):
        dre = DensityRatioEstimation(width)
        J_r = np.zeros(R)
        for r in range(R):
            Y_te_split = Y_te_arrays[:r] + Y_te_arrays[r+1:]
            dre.build(Y_rf, sum(Y_te_split, []))
            ratios = dre.compute_ratio_windows(Y_te_arrays[r])
            J_r[r] = np.mean(np.log(ratios))
        J[i] = np.mean(J_r)
    
    optimal_idx = np.nanargmax(J)
    return J, candidates[optimal_idx]

def ddre_ratios(Y,
                sigma_candidates,
                chunk_size,
                R,
                n_rf_te,
                build_args={},
                update_args={},
                tresh=None,
                verbose=False):
    """
    Computes ratios over all `Y` with shape (n, k, d)
    param `sigma_candidates` and `R` used in `kernel_sigma_selection`. Refer there for documentation
    param `chunk_size` is the size of chunk used in cross-validation
    param `n_rf_te` characterizes size of reference and test samples. They are equal due matrix multiplication
    by transposed itself (number of rows and columns must be equal)
    param `build_args` and `update_args` used in model building and parameter updating
    param `tresh` is treshold from original papaer 
    param `verbose` characterize whether to print progress every procent
    
    Returns:
    `ratios`
    `change_points` - indexes of changing. This is not empty if you specified right `tresh`
    """
    print('Finding optimal sigma...')
    J, optimal_sigma = kernel_sigma_selection(Y[:chunk_size * (2 * R - 1)], sigma_candidates, R)
    print(f'Optimal sigma is: {optimal_sigma}')

    n = len(Y)

    ratios = np.zeros(n)
    change_points = []
    t = n_rf_te + n_rf_te

    one_percent_size = math.ceil(n / 100)

    while t + 1 < n:
        dre = DensityRatioEstimation(optimal_sigma)
        dre.build(Y[t - n_rf_te - n_rf_te:t - n_rf_te], Y[t - n_rf_te:t], **build_args)

        while t + 1 < n:
            if verbose and (t % one_percent_size == 0):
                print(f'{t // one_percent_size}%')
            
            # numba can't compile next statement
            # dre.update_by_new_sample(Y[t], **update_args)
            # so hardcode this
            dre.update_by_new_sample(Y[t], update_args['lerning_rate'], update_args['reg_parameter'])


            ratios[t] = dre.compute_likelihood_ratio()
            t += 1
            if (tresh is not None) and (ratios[t - 1] > tresh):
                change_points.append(t - 1)
                t += n_rf_te + n_rf_te - 1
                break

    return ratios, change_points


def ddre_ratios_df(df, window_width, *args, **kwargs):
    """
    Forms "rolling windows" with size `window_width` for using by function `ddre_ratios`.
    For all needed arguments refer to function `ddre_ratios`
    """
    Y = rolling_window(df, window_width)
    return ddre_ratios(Y, *args, **kwargs)

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
    for candidate in width_candidates:
        other_params['window_width'] = candidate
        print(f'Candidate {candidate}')
        ratios, _ = ddre_ratios_df(Y, **other_params)

        abnormal_idxs, _ = find_peaks(ratios, distance=candidate)
        normal_idxs = inverse_ids(abnormal_idxs, ratios.shape[0])

        ssd = np.mean((ratios[abnormal_idxs] - ratios[normal_idxs].mean()) ** 2)
        ssds.append(ssd)
    return ssds, width_candidates[np.nanargmax(ssds)]
