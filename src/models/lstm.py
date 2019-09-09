from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, TimeDistributed
from keras.regularizers import l2

def build_model(input_length, input_shape, lstm_layers_size,
                reg_strength=0.01, dropout_coeff=0.1, **compile_attrs):
    """
    Builds lstm model with hidden layers of size `layers_size`.
    Returns values with shape (input_length, input_shape)
    """
    model = Sequential()

    model.add(LSTM(lstm_layers_size[0],
                   input_shape=(input_length, input_shape),
                   return_sequences=True, 
                   kernel_regularizer=l2(reg_strength)))

    for size in lstm_layers_size[1:]:
        # model.add(Dropout(dropout_coeff))
        model.add(LSTM(
            size,
            return_sequences=True,
            kernel_regularizer=l2(reg_strength),
        ))

    model.add(TimeDistributed(Dense(input_shape)))

    model.compile(loss="mse", **compile_attrs)
    return model
