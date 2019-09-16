import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def cross_validate(model_fn, n_splits, fit_params, X, y):
    tssplit = TimeSeriesSplit(n_splits=3)
    results = []

    for i, (train_idx, test_idx) in enumerate(tssplit.split(X, y)):
        print(f'Fold {i}...')
        model = model_fn()
        history = model.fit(X[train_idx],
                            y[train_idx],
                            validation_data=(X[test_idx], y[test_idx]),
                            **fit_params)
        results.append(history.history)
    
    return results

def rolling_window(data, window_length, shift=0):
    return np.array([
        data.iloc[i - window_length:i].to_numpy().flatten()
        for i in range(window_length + shift, data.shape[0])
    ]).reshape(-1, window_length, data.shape[1])

def inverse_ids(ids, rng):
    """
    Finds indexes that is not in `ids` in range [0, rng]
    """
    return [i for i in range(rng) if i not in ids]