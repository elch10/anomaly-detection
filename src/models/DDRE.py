# Implementation of Direct Density-Ratio Estimation (see referehces/kawahara2009.pdf)
import numpy as np
import pandas as pd
from numba import jitclass 
from numba import int32, float32
import math

from .utils import gaussian_kernel_function

# from multiprocessing import Pool, cpu_count
from src.features.build_features import rolling_window


def get_sequences_of_samples(Y, start_idx, n, k):
    if isinstance(Y, pd.DataFrame):
        Y = Y.iloc
    return rolling_window(Y[start_idx : start_idx + n + k], k)

def func_to_optimize(alphas, Y_te, sigma):
    ans = 0
    for i in range(Y_te.shape[0]):
        ans += np.log(sum([
            alphas[j] * gaussian_kernel_function(Y_te[i], Y_te[j], sigma)
            for j in range(Y_te.shape[0])
        ]))
    return -ans

def constrain(alphas, Y_rf, Y_te, sigma):
    n_te = Y_te.shape[0]
    n_rf = Y_rf.shape[0]
    return sum([
        sum([
            alphas[l] * gaussian_kernel_function(Y_rf[i], Y_te[l], sigma)
            for l in range(n_te)
        ])
        for i in range(n_rf)
    ]) / n_rf - 1


_spec = [
    ('sigma', float32),
    ('Y_rf', float32[:, :, :]),
    ('Y_te', float32[:, :, :]),
    ('k', int32),
    ('alphas', float32[:]),
    ('b', float32[:])
]

@jitclass(_spec)
class DensityRatioEstimation:
    def __init__(self, sigma):
        self.sigma = sigma
        self.Y_rf = np.array([[[0.]]], dtype=np.float32)
        self.Y_te = np.array([[[0.]]], dtype=np.float32)
        self.k = 0
        self.alphas = np.array([0.], dtype=np.float32)
        self.b = np.array([0.], dtype=np.float32)
    
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
        n_rf = self.Y_rf.shape[0]
        b = np.zeros(n_rf, dtype=np.float32)
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
        assert Y_rf.shape == Y_te.shape

        self.Y_rf = Y_rf.astype(np.float32)
        self.Y_te = Y_te.astype(np.float32)
        self.k = Y_te.shape[1]

        n_te = Y_te.shape[0]

        K = np.zeros((n_te, n_te), dtype=np.float32)
        for i in range(n_te):
            for l in range(n_te):
                K[i, l] = gaussian_kernel_function(Y_te[i], Y_te[l], self.sigma)

        self._compute_b()
        self.alphas = np.random.rand(n_te).astype(np.float32)

        for _ in range(iterations):
            prev_alphas = self.alphas.copy()

            # Perform gradient ascent
            temp = eps * np.dot(K.T, np.dot((1. / K).astype(np.float32), self.alphas))
            self.alphas += temp

            self._feasibility()

            if np.linalg.norm(self.alphas - prev_alphas) < min_delta:
                break

    def build_scipy(self, Y_rf, Y_te, tol=0.001):
        # assert Y_rf.shape == Y_te.shape

        self.Y_rf = Y_rf
        self.Y_te = Y_te
        self.k = Y_te.shape[1]

        n_te = Y_te.shape[0]
        n_rf = Y_rf.shape[0]

        from scipy.optimize import minimize

        res = minimize(
            fun=func_to_optimize, 
            x0=np.random.rand(n_te) + 0.001, 
            args=(Y_te, self.sigma),
            bounds=[(0, np.inf) for _ in range(n_te)],
            constraints=(
                dict(
                    type='eq',
                    fun=constrain,
                    args=(Y_rf, Y_te, self.sigma)
                )
            ),
            tol=tol,
        )
        
        if not res.success:
            raise ArithmeticError(res.message)
        self.alphas = res.x
        
    def compute_ratio_one_window(self, Y):
        res = np.zeros_like(self.alphas)
        for i in range(self.alphas.shape[0]):
            res[i] = gaussian_kernel_function(Y, self.Y_te[i], self.sigma)
        res *= self.alphas
        return np.sum(res)

    def compute_likelihood_ratio(self):
        return np.sum(np.log(self.compute_ratio_windows(self.Y_te)))
        
    def compute_ratio_windows(self, Y):
        """
        Y need to has shape (n_rf, k, d)
        """
        assert Y.shape[1] == self.k and Y.ndim == 3
        
        ratios = np.zeros(Y.shape[0], dtype=np.float32)

        for i in range(Y.shape[0]):
            ratios[i] = self.compute_ratio_one_window(Y[i])

        # with Pool(cpu_count()) as p:
        #     ratios = p.map(_helper, zip([self]*Y.shape[0], Y))

        return ratios
    
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
        new_alphas[-1] = lerning_rate / self.compute_ratio_one_window(y)
        self.alphas = new_alphas

        self.Y_rf[:-1] = self.Y_rf[1:]
        self.Y_rf[-1] = self.Y_te[0]
        
        self.Y_te[:-1] = self.Y_te[1:]
        self.Y_te[-1] = y

        self._compute_b()
        self._feasibility()

        
    
def _helper(args):
    return args[0].compute_ratio_one_window(args[1])


def kernel_width_selection(Y, candidates, R=4):
    """
    Selects optimal gaussian width.
    param `Y` is a "rolling" window. It musts have shape of (n, k, d)
    param `R` characterize the size of each split chunk. The `n` must be divisible by `2 * R - 1`
    The first chunk would be used as reference sample. And others `R-1` as test in cross-validation
    """
    n = Y.shape[0]
    assert n % (2 * R - 1) == 0

    chunk_size = n // (2 * R - 1)
    
    Y_rf = Y[:(R - 1) * chunk_size]
    Y_te_copy = np.copy(Y[(R - 1) * chunk_size:])
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
    
    optimal_idx = np.nanargmax(J)
    return J, candidates[optimal_idx]

def ddre_ratios(Y,
                sigma_candidates,
                n_sigma,
                R,
                n_rf_te,
                build_args={},
                update_args={},
                tresh=None,
                verbose=False):
    """
    Computes ratios over all `Y` with shape (n, k, d)
    param `sigma_candidates` and `R` used in `kernel_width_selection`. Refer there for documentation
    param `n_sigma` is the number of first examples used to find optimal sigma
    param `n_rf_te` characterizes size of reference and test samples. They are equal due matrix multiplication
    by transposed itself (number of rows and columns must be equal)
    param `verbose` characterize whether to print progress every procent
    param
    
    Returns:
    `ratios`
    `change_points` - indexes of changing. This is not empty if you specified right `tresh`
    """
    print('Finding optimal sigma...')
    J, optimal_sigma = kernel_width_selection(Y[:n_sigma], sigma_candidates, R)
    print(f'Optimal sigma is: {optimal_sigma}')

    n = Y.shape[0]

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
            
            # numba can't compile this
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


def ddre_ratios_df(df, k, *args, **kwargs):
    """
    Forms "rolling windows" with size `k` for using by function `ddre_ratios`.
    For all needed arguments refer to function `ddre_ratios`
    """
    Y = rolling_window(df, k)
    return ddre_ratios(Y, *args, **kwargs)
