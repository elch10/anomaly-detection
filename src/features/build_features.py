import numpy as np
import pandas as pd

def rolling_window(data, window_length, shift=0):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    return np.array([
        np.ndarray.view(data[i - window_length:i]).flatten()
        for i in range(window_length + shift, data.shape[0])
    ]).reshape(-1, window_length, data.shape[1])