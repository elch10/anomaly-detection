#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cufflinks as cf
from src.features.build_features import rolling_window
from densratio import densratio


#%%
def compute_scores_by_densratio(Y, n_rf, n_te, densratio_args):
    scores = np.zeros(Y.shape[0])

    for i in range(Y.shape[0] - n_rf - n_te):
        if i % 50 == 0:
            print(i)
        densratio_obj = densratio(Y[i : i + n_rf], Y[i + n_rf: i + n_rf + n_te], **densratio_args)
        scores[i+n_rf] = np.sum(densratio_obj.compute_density_ratio(Y[i + n_rf: i + n_rf + n_te]))
        break
    return scores



#%%
n_rf = 10
n_te = 10
k = 16

densratio_args = dict(
    alpha = 0.1,
    sigma_range=[10],
    lambda_range=[1],
    kernel_num=n_rf, 
    verbose=False,
)


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
Y = rolling_window(y, k)

# %%
%%time

scores = compute_scores_by_densratio(Y, n_rf, n_te, **densratio_args)
pd.DataFrame(dict(scores=scores)).iplot()

# %%














# %%
y = [0, 8, 6, 4]

for i in range(4, 1000):
    y.append(0.97 * y[i-1] + 0.4 * y[i - 2] - 0.5 * y[i-3] + 0.2 * np.random.randn())
for i in range(1000, 1500):
    y.append(0.97 * y[i-1] + 0.4 * y[i - 2] - 0.8 * y[i-3] + 0.2 * np.random.randn())

true_change_points = [1000]
plt.plot(y)


# %%
y = np.array(y).reshape(-1, 1)
Y = rolling_window(y, k)

# %%

%%time
scores = compute_scores_by_densratio(Y, n_rf, n_te, densratio_args)
pd.DataFrame(dict(scores=scores)).iplot()


# %%

n_rf = 100
n_te = 100
k = 50

densratio_args = dict(
    alpha = 0.1,
    # sigma_range=[10],
    # lambda_range=[1],
    kernel_num=n_rf, 
    verbose=True,
)


data = pd.read_csv('data/processed/tep_data.csv', index_col='Index')
print(f'Len of dataset: {data.shape[0]}')

scores = compute_scores_by_densratio(rolling_window(data, k), n_rf, n_te, densratio_args)
pd.DataFrame(dict(scores=scores)).iplot()