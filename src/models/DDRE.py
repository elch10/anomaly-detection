# Implementation of Direct Density-Ratio Estimation (see referehces/kawahara2009.pdf)

from scipy.linalg import norm
import numpy as np

class GaussianKernel:
    def __init__(self, sigma, center):
        self.sigma = sigma
        self.center = center
    
    def denstity_at(self, X):
        """
        Computes density at X
        """
        return np.exp(-norm(X - self.center)**2 / (2 * self.sigma**2))

class DensityRatioEstimation:
    def __init__(self, alphas, sigma):
        """
        `alphas` must be equal to the `n_test`
        """
        self.alphas = alphas
        self.sigma = sigma
    
    def build(self, Y_refrence, Y_test):
        """

        """

