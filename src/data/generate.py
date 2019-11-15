import numpy as np
import pandas as pd
from itertools import product

def generate_anomalies(length):
    """
    Generates different types of one dimensional anomalies with length `window_length`
    """
    anomalies = [
        np.zeros(length), *[
            np.random.randint(1, 4) * np.sin(k * np.linspace(0, 2 * np.pi, num=length))
            for k in range(1, 20)
        ]
    ]
    return np.array(anomalies)


def insert_anomalies(X, amount, axis=None, anomaly_length=None):
    """
    Inserts anomalies in copy of `X` to the specified `axis`. If `axis` is `None`, then last dimension will be choosed
    param `amount` characterizes amount of anomalies needed to insert in `X`
    param `anomaly_length` - length of anomaly. If it is `None` then anomaly will be inserted at whole `axis`
    Returns `X` with anomalies inserted, and indexes of starting points of anomalies where they was inserted
    """
    assert len(X.shape) > 1 and anomaly_length <= X.shape[axis if axis is not None else -1]

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    else:
        X = X.copy()
    
    if axis != None:
        X = np.swapaxes(X, len(X.shape)-1, axis)

    list_to_product = [range(x) for x in X.shape[:-1]] + [range(X.shape[-1] - anomaly_length + 1)]

    if anomaly_length is None:
        anomaly_length = X.shape[-1]

    anomalies = generate_anomalies(anomaly_length)
    
    all_idxs = np.array(list(product(*list_to_product)))
    anom_idxs = np.random.choice(len(all_idxs), amount, replace=False)
    anom_types = np.random.choice(len(anomalies), amount)

    for i, idx in enumerate(anom_idxs):
        X[tuple(all_idxs[idx, :-1]) + (slice(all_idxs[idx, -1], all_idxs[idx, -1] + anomaly_length), )] = anomalies[anom_types[i]]

    idxs = all_idxs[anom_idxs]
    if axis != None:
        X = np.swapaxes(X, len(X.shape)-1, axis)
        idxs[:, [len(X.shape)-1, axis]] = idxs[:, [axis, len(X.shape)-1]]

    return X, idxs
