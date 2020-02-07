# Implementation of Direct Density-Ratio Estimation (see referehces/kawahara2009.pdf)
import math
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import numba as nb
from sklearn.preprocessing import StandardScaler

from src.utils import inverse_ids
from src.features.build_features import rolling_window

@nb.jit(nopython=True)
def gaussian_kernel_function(Y1, Y2, sigma):
    return np.exp(-np.linalg.norm(Y1 - Y2)**2 / (2 * sigma**2))


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
def kernel_sigma_selection(df, window_width, candidates, chunk_size, R=3):
    """
    Selects optimal gaussian width (standard deviation).
    param `df` is a dataframe with shape (n, d)
    param `window_width` is the setted rolling window size
    param `candidates` is the candidates from which will be found optimal sigma
    param `chunk_size` is the size of chunk used in cross-validation
    param `R` characterize the number of splits in cross_validation
    The first chunk would be used as reference sample. And others `R-1` as test in cross-validation
    """
    if len(candidates) == 1:
        return np.zeros(0, dtype=np.float64), candidates[0]

    needed_n = chunk_size * (2 * R - 1) + 2 * window_width

    assert needed_n <= df.shape[0]
    df = df.copy()[:needed_n]
    
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
    print('Optimal sigma is:', candidates[optimal_idx])
    return J, candidates[optimal_idx]

@nb.njit(parallel=True)
def ddre_ratios(df,
                window_width,
                sigma,
                n_rf_te=32,
                eps=0.001, min_delta=0.01, iterations=100,
                learning_rate=1, reg_parameter=0.01,
                tresh=-1,
                verbose=False,
                chunks=-1):
    """
    Computes ratios over all `df` with shape (n, d). It need to be the numpy array, not pd.DataFrame!
    param `window_width` is a width of rolling window
    param `sigma` is the standard deviation of gaussian kernel function
    param `n_rf_te` characterizes the number of reference and test samples
    param `eps`, `min_delta`, `iteratios` used in DDRE building
    param `learning_rate`, `reg_parameter` used in parameter updating of DDRE model
    param `tresh` is treshold from original paper (if it is -1, then it is ignored)
    param `verbose` characterize whether to print progress every 5%
    param `chunks` is the number of threads used for computation. If it is equal to `-1` then amount of chunks would be seleceted automaticly
    
    Returns:
    `ratios` - is a computed ratios of probability densities
    `change_points` - indexes of changing. This is not empty if you specified right `tresh`
    """
    n = df.shape[0]

    ratios = np.zeros(n, dtype=np.float64)
    change_points = []

    chunks_ = cpu_cnt

    piece_size = n // chunks_
    pieces_cnt = n // piece_size

    # set minimal piece length to two minimal lengths needed for DDRE
    minimal_length = 2 * (n_rf_te * 2 + window_width * 2)

    while piece_size <= minimal_length and chunks_ > 1:
        chunks_ //= 2

        piece_size = n // chunks_
        pieces_cnt = n // piece_size
    
    if chunks > 0 and (n // chunks) >= minimal_length:
        piece_size = n // chunks
        pieces_cnt = chunks


    five_percent_size = math.ceil(piece_size / 20)
    print('Computing ratios in parrallel with', pieces_cnt, 'chunks')

    for i in nb.prange(pieces_cnt):
        start_left = i*piece_size+n_rf_te*2+window_width*2
        t = start_left

        if i == pieces_cnt - 1:
            right = n - 1
        else:
            right = (i+1)*piece_size

        while t < right:
            dre = DensityRatioEstimation(sigma, window_width, n_rf_te)
            dre.build(df[i*piece_size:right+1], eps, min_delta, iterations)

            while t < right:
                if verbose and ((t - i * piece_size) % five_percent_size == 0):
                    print(i, 5 * (t - i * piece_size) // five_percent_size, '%')
                
                dre.update_by_next_sample(learning_rate, reg_parameter)

                ratios[t] = dre.compute_likelihood_ratio()
                t += 1
                if tresh != -1 and ratios[t - 1] > tresh:
                    change_points.append(t - 1)
                    t += n_rf_te + n_rf_te - 1
                    break

        ratios[i*piece_size: start_left] = np.mean(ratios[start_left+1:right])

    return ratios, change_points

def find_abnormal_idxs(ratios, window_width, amount=3):
    abnormal_idxs, _ = find_peaks(ratios, distance=window_width)
    abnormal_idxs = sorted(abnormal_idxs, key=lambda idx:abs(ratios[idx]))[-amount:]
    return abnormal_idxs

def kernel_width_selection(Y, search_params, DDRE_params):
    """
    Finds appropriate width of rolling window
    param `Y` - data with shape (n, d)
    param `additional_params` - params for finding window width and sigma
    param `DDRE_params` - params that would be passed to `ddre_ratios` function

    Returns tuple of elements:
    1. Sum squared distance beetwen mean of non change-points derivatives of ratios and
    change-points derivatives of ratios for every width candidate in `width_candidates`
    2. Optimal width that corresponds to maximal sum squared distance
    3. Optimal sigma
    """
    ssds = []
    optimal_sigmas = []
    width_candidates = search_params['width_candidates']

    sigma_candidates = search_params['sigma_candidates']
    chunk_size = search_params['chunk_size']
    R = search_params.get('R', 3)

    if len(width_candidates) == 1:
        _, optimal_sigma = kernel_sigma_selection(Y, width_candidates[0], sigma_candidates,
                                                  chunk_size, R)
        return ssds, width_candidates[0], optimal_sigma

    for candidate in width_candidates:
        DDRE_params['window_width'] = candidate
        print('Candidate', candidate)
        
        _, optimal_sigma = kernel_sigma_selection(Y, candidate, sigma_candidates,
                                                  chunk_size, R)
        DDRE_params['sigma'] = optimal_sigma
        optimal_sigmas.append(optimal_sigma)

        n = chunk_size * (2 * R - 1) + 2 * candidate
        ratios, _ = ddre_ratios(Y[:n], **DDRE_params)

        abnormal_idxs = find_abnormal_idxs(ratios, candidate)
        normal_idxs = inverse_ids(abnormal_idxs, ratios.shape[0])

        scaled = StandardScaler().fit_transform(ratios[ratios.nonzero()[0], None]).ravel()
        ratios[ratios.nonzero()[0]] = scaled

        abnormal = ratios[abnormal_idxs]

        ssd = np.nanmean((abnormal[~np.isnan(abnormal)] - np.nanmean(ratios[normal_idxs])) ** 2)
        ssds.append(ssd)

    return ssds, width_candidates[np.nanargmax(ssds)], optimal_sigmas[np.nanargmax(ssds)]


def compute_ratios(y, search_params, DDRE_params):
    """
    Finds density ratios with hyperparameter search.
    Optimal values will be added to DDRE_params dictionary.
    """
    print('Finding hyperparams...')
    _, window_width, sigma = kernel_width_selection(y, search_params, DDRE_params)
    print(f'\nOptimal width is {window_width}\n')

    DDRE_params['window_width'] = window_width
    DDRE_params['sigma'] = sigma
    
    print('Starting compute ratios...')
    ratios, chng_pts = ddre_ratios(y, **DDRE_params)
    peaks = find_abnormal_idxs(ratios, window_width)
    
    return ratios, chng_pts, peaks
