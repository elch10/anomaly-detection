from sklearn.model_selection import TimeSeriesSplit

import numpy as np

from itertools import product

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

def inverse_ids(ids, rng):
    """
    Finds indexes that is not in `ids` in range [0, rng]
    """
    return [i for i in range(rng) if i not in ids]


def generate_anomalies(length):
    """
    Generates different types of one dimensional anomalies with length `window_length`
    """
    anomalies = [
        np.zeros(length), *[
            np.random.randint(2, 4) * np.sin(k * np.linspace(0, 2 * np.pi, num=length))
            for k in range(1, 20)
        ]
    ]
    return np.array(anomalies)


def insert_anomalies(X, amount, axis=None):
    """
    Inserts anomalies in `X` to the specified `axis`. If `axis` is None, then last dimension will be choosed
    Returns indexes where anomalies was inserted
    """
    assert len(X.shape) > 1
    
    if axis != None:
        X = np.swapaxes(X, len(X.shape)-1, axis)

    anomalies = generate_anomalies(X.shape[-1])

    all_idxs = np.array(list(product(*[range(x) for x in X.shape[:-1]])))
    anom_idxs = np.random.choice(len(all_idxs), amount, replace=False)
    anom_types = np.random.choice(len(anomalies), amount)
    
    X[list(zip(*all_idxs[anom_idxs]))] = anomalies[anom_types]

    if axis != None:
        X = np.swapaxes(X, len(X.shape)-1, axis)

    return X, all_idxs[anom_idxs]