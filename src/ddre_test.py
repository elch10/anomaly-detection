# %%
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..'))
	print(os.getcwd())
except:
	pass


#%%
# import importlib
# import src.models.DDRE
# importlib.reload(src.models.DDRE)

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cufflinks as cf
from src.features.build_features import rolling_window
from src.models.DDRE import kernel_width_selection, DensityRatioEstimation, get_sequences_of_samples, ddre_ratios_df, kernel_width_selection


#%%
# Lets check algorithm on synthetic dataset
n = 1000
mean = 0
step = 100
y = np.zeros(n, dtype=float)
for i in range(2, n):
    if i % step == 0:
        mean = mean + i // step
    y[i] = y[i-1]*0.6 - y[i-2]*0.5 + mean + np.random.randn()

plt.plot(y)
plt.vlines(list(range(0, n, step)), np.min(y), np.max(y), linestyle='--', colors='r')


#%%
params = dict(k=32,
              sigma_candidates=[0.1, 1, 5, 10, 20, 50, 100, 1000],
              n_sigma=15 * 7,
              R=4,
              n_rf_te=15,
              verbose=True,
              build_args=dict(),
              update_args=dict(lerning_rate=1, reg_parameter=0.01)
)

ratios, chng_pts = ddre_ratios_df(y.reshape(-1, 1), **params)

# %%

pd.DataFrame(dict(ratios=ratios)).iplot()



#%%
import importlib
import src.data.generate
importlib.reload(src.data.generate)

from src.data.generate import insert_anomalies

# %%
data = pd.read_csv('data/processed/tep_data.csv', index_col='Index')
print(f'Len of dataset: {data.shape[0]}')

# %%
changed_df = insert_anomalies(data, 50, axis=0, anomaly_length=50)