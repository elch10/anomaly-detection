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

def inverse_ids(ids, rng):
    """
    Finds indexes that is not in `ids` in range [0, rng]
    """
    return [i for i in range(rng) if i not in ids]


def find_anomaly(differences, treshold):
    return np.where(differences > treshold)[0]

def compute_diff(model, X, y):
    prediction = model.predict(X)
    return np.abs(y-prediction).sum(axis=2).sum(axis=1)

def recall_all_tresh(window_diffs, anom_idxs, window_length):
    recalls = []
    pred_anom_segment = np.zeros(len(window_diffs), dtype=int)

    treshs = list(reversed(np.sort(window_diffs)))

    for tresh in treshs:
        start_idx = np.where(window_diffs == tresh)[0][0]
        pred_anom_segment[start_idx:start_idx + window_length] = 1

        recall = (
            (anom_idxs == pred_anom_segment) &
            (anom_idxs == 1)).sum() / anom_idxs.sum()
        recalls.append(recall)

    return treshs, recalls

from sklearn.metrics import recall_score
def intersection_over_true(max_len, anom_idxs, anom_lens, pred_idxs, prediction_len):
    """
    Recall
    """
    really_anom_segment = np.zeros(max_len)
    for left, l in zip(anom_idxs, anom_lens):
        really_anom_segment[left:left+l] = 1
    
    pred_anom_segment = np.zeros(max_len)
    for left in pred_idxs:
        pred_anom_segment[left:left+prediction_len] = 1

    return recall_score(really_anom_segment, pred_anom_segment)