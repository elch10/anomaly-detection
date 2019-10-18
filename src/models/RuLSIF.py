from .utils import gaussian_kernel_function
import numpy as np

class RuLSIF:
    def __init__(self, alpha, sigma):
        self.theta = None
        self.Y_prime = None
        self.Y = None
        self.alpha = alpha
        self.sigma = sigma

    def build(self, Y_prime, Y, reg_term=0.01):
        """
        param `Y_prime` is analogue of reference samples from article 
        `Change-Point Detection in Time-Series Databy Direct Density-Ratio Estimation 2009`
        param `Y_prime` and `Y` would have shape (n, k, d)
        param `reg_term` used for regularization
        """
        assert Y.shape == Y_prime.shape

        self.Y_prime = Y_prime
        self.Y = Y

        n = Y.shape[0]

        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                H[i, j] = (1 - self.alpha) * np.mean([
                    gaussian_kernel_function(Y_prime[k], Y[i], self.sigma)*\
                    gaussian_kernel_function(Y_prime[k], Y[j], self.sigma)
                    for k in range(n)
                ])

                H[i, j] += self.alpha * np.mean([
                    gaussian_kernel_function(Y[k], Y[i], self.sigma)*\
                    gaussian_kernel_function(Y[k], Y[j], self.sigma)
                    for k in range(n)
                ])
        
        h = np.zeros(n)
        for i in range(n):
            h[i] = np.mean([
                gaussian_kernel_function(Y[j], Y[i], self.sigma)
                for j in range(n)
            ])

        self.theta = np.linalg.inv(H + reg_term * np.identity(n)).dot(h)

    def _g(self, Y):
        """
        param `Y` need to has shape (k, d)
        """
        assert Y.shape == self.Y.shape[1:]

        return np.sum(list(map(lambda pair: pair[0] * gaussian_kernel_function(Y, pair[1], self.sigma), 
                           zip(self.theta, self.Y))))
        
    def compute_change_score(self):
        n = self.Y.shape[0]
        f = lambda el: self._g(el) ** 2

        first_term = -self.alpha / (2 * n) * sum(list(map(f, self.Y)))
        second_term = -(1 - self.alpha) / (2 * n) * sum(list(map(f, self.Y_prime)))
        third_term = np.mean(list(map(self._g, self.Y)))

        return first_term + second_term + third_term - 1/2

