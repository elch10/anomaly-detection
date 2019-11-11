import numpy as np
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
    Inserts anomalies in `X` to the specified `axis`. If `axis` is `None`, then last dimension will be choosed
    param `amount` characterizes amount of anomalies needed to insert in `X`
    param `anomaly_length` - length of anomaly. If it is `None` then anomaly will be inserted at whole `axis`
    Returns X with anomalies inserted, and indexes where anomalies was inserted
    """
    assert len(X.shape) > 1 and anomaly_length <= X.shape[axis if axis is not None else -1]
    
    if axis != None:
        X = np.swapaxes(X, len(X.shape)-1, axis)

    list_to_product = [range(x) for x in X.shape[:-1]]

    if anomaly_length is None:
        anomaly_length = X.shape[-1]
    else:
        list_to_product += [range(X.shape[-1] - anomaly_length)]

    anomalies = generate_anomalies(anomaly_length)
    
    all_idxs = np.array(list(product(*list_to_product)))
    anom_idxs = np.random.choice(len(all_idxs), amount, replace=False)
    anom_types = np.random.choice(len(anomalies), amount)

    for i, idx in enumerate(anom_idxs):
        X[tuple(all_idxs[idx, :-1]) + (slice(all_idxs[idx, -1], all_idxs[idx, -1] + anomaly_length), )] = anomalies[anom_types[i]]

    # X[tuple(zip(*all_idxs[anom_idxs, :-1]))+slices] = anomalies[anom_types]

    if axis != None:
        X = np.swapaxes(X, len(X.shape)-1, axis)

    return X, all_idxs[anom_idxs]
