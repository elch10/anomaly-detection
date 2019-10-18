import numpy as np

def gaussian_kernel_function(Y1, Y2, sigma):
    return np.exp(-np.linalg.norm(Y1 - Y2)**2 / (2 * sigma**2))
