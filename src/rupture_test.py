#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
#%%
n = 300
mean = 0
step = 30
true_change_points = []
y = np.zeros(n, dtype=float)
for i in range(2, n):
    if i % step == 0:
        mean += 1
        true_change_points.append(i)
    
    y[i] = y[i-1]*0.6 - y[i-2]*0.5 + mean + np.random.randn()

plt.plot(y)
plt.vlines(list(range(0, n, step)), -2, 15, linestyle='--', colors='r')

#%%
y = y.reshape(-1, 1)
model = 'rbf'
algo = rpt.Pelt(model="rbf", min_size=10, ).fit(y)
result = algo.predict(3)

rpt.display(y, true_change_points, result)


#%%
data = pd.read_csv('data/processed/tep_data.csv', index_col='Index')
print(f'Len of dataset: {data.shape[0]}')

#%%
%%time
first_examples_amount = 1000
signal = data.to_numpy()[:first_examples_amount]
model = 'rbf'
algo = rpt.Pelt(model="rbf", min_size=10, ).fit(signal)
result = algo.predict(pen=10)
rpt.display(signal, result, result)

