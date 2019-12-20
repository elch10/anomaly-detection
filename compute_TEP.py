import numpy as np
import pandas as pd
import cufflinks as cf
from src.models.DDRE import *

params = dict(
    sigma_candidates=np.array([10, 20, 50, 100, 200, 500], dtype=np.float32),
    chunk_size=500,
    R=3,
    n_rf_te=50,
    verbose=True,
    # build args
    eps=0.001, min_delta=0.01, iterations=100,
    # update args
    learning_rate=1, reg_parameter=0.01,
)

data = pd.read_csv('data/processed/TEP.csv', index_col='Index')
print(f'Len of dataset: {data.shape[0]}')


ratios, chng_pts, peaks = compute_ratios(data.to_numpy(), [200, 500, 750, 1000], params, ratio=0.05)

np.save('TEP_ratios.npy', ratios)
np.save('TEP_chng_pts.npy', chng_pts)
np.save('TEP_peaks.npy', peaks)