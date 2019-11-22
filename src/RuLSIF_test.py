#%%
import importlib
import src.models.RuLSIF
importlib.reload(src.models.RuLSIF)

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.features.build_features import rolling_window
from src.models.RuLSIF import RuLSIF

#%%
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
y = y.reshape(-1, 1)
k = 16
Y = rolling_window(y, k)

#%%
%%time
params = dict(
    alpha=0,
    sigma=10,
)
rulsif = RuLSIF(**params)

n_rf = 10
n_te = 10

scores = np.zeros(Y.shape[0])
for i in range(Y.shape[0] - n_rf - n_te):
    if i % 50 == 0:
        print(i)
    rulsif.build(Y[i : i + n_rf], Y[i + n_rf: i + n_rf + n_te])
    scores[i+n_rf] = rulsif.compute_change_score()

#%%
import cufflinks
pd.DataFrame(dict(scores=scores)).iplot()

#%%
