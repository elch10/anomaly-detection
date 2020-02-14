import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, TimeDistributed
from keras.regularizers import l2

from ..data.generate import insert_anomalies
from ..features.build_features import rolling_window

def lstm_model(input_length, input_shape, lstm_layers_size, loss='mae',
                reg_strength=0.01, dropout_coeff=0.1, **compile_attrs):
    """
    Builds lstm model with hidden layers of size `layers_size`.
    Returns values with shape (input_length, input_shape)
    """
    model = Sequential()

    model.add(LSTM(
        lstm_layers_size[0],
        input_shape=(input_length, input_shape),
        return_sequences=True, 
        kernel_regularizer=l2(reg_strength)
    ))

    for size in lstm_layers_size[1:]:
        if 0 < dropout_coeff < 1:
            model.add(Dropout(dropout_coeff))
        
        model.add(LSTM(
            size,
            return_sequences=True,
            kernel_regularizer=l2(reg_strength),
        ))

    # For equivalent transformation `embeddings` to real values
    model.add(TimeDistributed(Dense(input_shape, kernel_regularizer=l2(reg_strength))))
    model.compile(loss, **compile_attrs)
    return model

def find_anomaly(differences, treshold):
    return np.where(differences > treshold)[0]


def fit_generator(X, y, batch_size=64):
    while True:
        idxs = np.random.randint(len(X), size=batch_size)
        yield np.take(X, idxs, axis=0), np.take(y, idxs, axis=0)

def predict_generator(X, batch_size=64):
    for i in range(0, len(X), batch_size):
        yield np.array(X[i:i+batch_size])

def compute_diff(model, X, y):
    prediction = model.predict_generator(predict_generator(X), steps=(len(X)+64-1)//64)
    return np.abs(y-prediction).sum(axis=2).sum(axis=1)

def recall_of_tresh(window_diffs, true_idxs, window_length):
    recalls = []
    pred_anom_segment = np.zeros(len(window_diffs), dtype=int)

    treshs = list(reversed(np.sort(window_diffs)))

    for tresh in treshs:
        start_idx = np.where(window_diffs == tresh)[0][0]
        pred_anom_segment[start_idx:start_idx + window_length] = 1

        recall = (
            (true_idxs == pred_anom_segment) &
            (true_idxs == 1)).sum() / true_idxs.sum()
        recalls.append(recall)

    return treshs, recalls

def intersection_over_true(max_len, anom_start_idxs, anom_lens, pred_idxs, window_length, score_function):
    """
    Recall
    """
    really_anom_segment = np.zeros(max_len)
    for left, l in zip(anom_start_idxs, anom_lens):
        really_anom_segment[left:left+l] = 1
    
    pred_anom_segment = np.zeros(max_len)
    for left in pred_idxs:
        pred_anom_segment[left:left+window_length] = 1

    return score_function(really_anom_segment, pred_anom_segment)

def find_optimal_tresh(model, X, y, anoms_dataset_shape, window_length, prediction_len, plot=False):
    """
    param `X` and `y` is the normal data, on which `model` was trained
    """
    anom_dataset = np.random.randn(*anoms_dataset_shape)
    anom, _ = insert_anomalies(anom_dataset, np.prod(anoms_dataset_shape[1:]) if len(anoms_dataset_shape) > 1 else 1, axis=0)

    X_anom = rolling_window(anom, window_length)[:-prediction_len]

    anom_diffs = compute_diff(model, X_anom, y)
    anom_treshs, anom_recalls = recall_of_tresh(anom_diffs, np.ones_like(anom_diffs), window_length)

    normal_diffs = compute_diff(model, X, y)
    norm_recalls = []

    for tresh in anom_treshs:
        recall = (normal_diffs < tresh).sum() / normal_diffs.shape[0]
        norm_recalls.append(recall)

    optimum = anom_treshs[np.argmin(np.abs(np.array(anom_recalls) - np.array(norm_recalls)))]
    
    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        plt.plot(anom_treshs, anom_recalls, label='Только сгенерированные аномалии')
        plt.plot(anom_treshs, norm_recalls, label='Только нормальные данные')

        plt.vlines(optimum, 0, 1, linestyles='dashed', label='Оптимум')

        plt.title(f'Оптимум - {optimum:.2f}')
        plt.xlabel('Порог')
        plt.ylabel('Доля корректных')
        plt.legend()
        plt.show()
    
    return optimum