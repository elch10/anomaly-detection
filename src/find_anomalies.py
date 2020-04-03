import numpy as np
import pandas as pd
from src.models.DDRE import *

from src.models.lstm import *
import keras

import time
timestr = time.strftime("%Y%m%d-%H%M%S")

data = pd.read_csv('data/processed/example.csv', index_col='Index')
print(f'Len of dataset: {data.shape[0]}')

DDRE_search_params = dict(
    width_candidates=[50, 60, 70, 80, 100],
    sigma_candidates=np.array([0.1, 0.5, 1, 5, 10, 20, 50], dtype=np.float32),
    chunk_size=100,
    R=3,
)

DDRE_params = dict(
    n_rf_te=50,
    verbose=True,
    # build args
    eps=0.001, min_delta=0.01, iterations=100,
    # update args
    learning_rate=1, reg_parameter=0.01,
)

ratios, chng_pts, peaks = compute_ratios_hp(data.to_numpy(), DDRE_search_params, DDRE_params)

np.save('result/ratios-%s.npy' % (timestr, ), ratios)
np.save('result/chng_pts-%s.npy' % (timestr, ), chng_pts)
np.save('result/peaks-%s.npy' % (timestr, ), peaks)


window_length=64
prediction_len=2

LSTM_build_params = dict(
    input_length=window_length,
    input_shape=data.shape[1],
    lstm_layers_size=[32, 32],
    dropout_coeff=0.1,
    optimizer=keras.optimizers.Adam(lr=0.01),
    reg_strength=0.005,
)

batch_size = 64

LSTM_fit_params = dict(
    epochs=20,
    verbose=1,
    callbacks=[keras.callbacks.ReduceLROnPlateau(patience=3),
               keras.callbacks.EarlyStopping(min_delta=0.01, patience=2)]
)

boundaries = np.concatenate(([0], peaks, [data.shape[0]]))
anom_idxs = []

for i, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
    if right - left < window_length + prediction_len:
        print(f'Piece {i} from {left} to {right} is too small. Continuing without current piece...')
        continue
    
    print(f'Piece {i} from {left} to {right}...')
    piece = data.iloc[left:right]
    X = rolling_window(piece, window_length, 0)[:-prediction_len]
    y = rolling_window(piece, window_length, prediction_len)
    
    model = lstm_model(**LSTM_build_params)
    b_size = min(len(y), batch_size)
    model.fit_generator(fit_generator(X, y, b_size), **LSTM_fit_params, steps_per_epoch=max(int(0.2 * len(X) / b_size), 1))
    
    tresh = find_optimal_tresh(model, X, y, piece.shape, window_length, prediction_len, True)
    start_idxs = find_anomaly(compute_diff(model, X, y), tresh)
    for i in start_idxs:
        anom_idxs.extend(range(i+left, i+left+window_length))

np.save('result/anom_idxs-%s.npy' % (timestr, ), anom_idxs)
