from keras import Sequential
from keras.layers import LSTM, Dropout, Dense

def build_model(input_length, input_shape, layers_size, dropout_coeff=0.2):
    """
    Builds lstm model with hidden layers of size `layers_size`
    """
    model = Sequential()
    # layers = {'input': 1, 'hidden1': 64, 'hidden2': 256, 'hidden3': 100, 'output': 1}

    model.add(LSTM(layers_size[0],
            input_shape=(input_length, input_shape),
            return_sequences=True))

    for i, size in enumerate(layers_size, 1):
        model.add(Dropout(dropout_coeff))
        model.add(LSTM(
            size,
            return_sequences=True,
        ))

    model.compile(loss="mse", optimizer="rmsprop")
    return model
