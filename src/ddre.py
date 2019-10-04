#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.DDRE import kernel_width_selection, DensityRatioEstimation, get_sequences_of_samples

#%%
data = pd.read_csv('data/processed/tep_data.csv', index_col='Index')
print(f'Len of dataset: {data.shape[0]}')

#%%
import importlib
import src.models.DDRE
importlib.reload(src.models.DDRE)


#%%
%%time


sigmas = [2, 4, 5, 6, 8, 10, 15]
n_rf = 16
n_te = 24
k = 8

Y_re = get_sequences_of_samples(data, 0, n_rf, k)
Y_te = get_sequences_of_samples(data, n_rf, n_te, k)

optimal_sigma = kernel_width_selection(Y_re, Y_te, sigmas, 3)
print(f'Optimal sigma is: {optimal_sigma}')

#%%
from src.models.DDRE import DensityRatioEstimation, get_sequences_of_samples
n_rf=16
n_te=16
k=8

Y_re = get_sequences_of_samples(data, 0, n_rf, k)
Y_te = get_sequences_of_samples(data, n_rf, n_te, k)

dre = DensityRatioEstimation(optimal_sigma)

dre.build(Y_re, Y_te)

ratios = dre.compute_ratios(data)
plt.hist(ratios)
# %%time
plt.loglog(ratios)


#%%
# Lets check algorithm on synthetic dataset
synthetic_data = np.sin(np.linspace(0, 2 * np.pi, num=40))
synthetic_data = np.append(synthetic_data, np.zeros(60)).reshape(-1, 1)

#%%
n_rf=16
n_te=24
k=8

sigmas = [1, 2, 4, 5, 6, 8, 10, 15]

Y_re = get_sequences_of_samples(synthetic_data, 0, n_rf, k)
Y_te = get_sequences_of_samples(synthetic_data, n_rf, n_te, k)

optimal_sigma = kernel_width_selection(Y_re, Y_te, sigmas, 3)

#%%
optimal_sigma = 1
n_rf=16
n_te=16
k=8

Y_re = get_sequences_of_samples(synthetic_data, 0, n_rf, k)
Y_te = get_sequences_of_samples(synthetic_data, n_rf, n_te, k)

dre = DensityRatioEstimation(optimal_sigma)
dre.build(Y_re, Y_te)
ratios = dre.compute_ratios(synthetic_data)

plt.plot(ratios)

#%%


#%%
