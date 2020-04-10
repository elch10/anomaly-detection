from keras import Sequential
from keras.layers import Dense, Input, Dropout
from keras.regularizers import l2
from keras.models import Model
import keras
import numpy as np

def create_autoencoder(input_shape, hidden_layer_size, 
                       hidden_layer_activation=None,
                       reg_strength=0.01, input_dropout=0.1):
    """
    Creates autoencoder model with hidden layer of size `hidden_layer_size`
    If `input_dropout` is in range [0, 1] then would be added Dropout on input data with `input_dropout` rate
    """
    model = Sequential()

    if 0 < input_dropout < 1:
        model.add(Dropout(input_dropout))

    model.add(Dense(
        hidden_layer_size,
        input_shape=(input_shape,),
        activation=hidden_layer_activation,
        kernel_regularizer=l2(reg_strength)
    ))

    model.add(Dense(input_shape, kernel_regularizer=l2(reg_strength)))
    return model


def build_autoencoder(create_params, compile_params):
    """
    Creates and builds autoencoder model with `create_params` used for call `create_autoencoder`
    If `compile_params` has not `loss` then `mae` would be used for training
    """
    model = create_autoencoder(**create_params)

    compile_params['loss'] = compile_params.get('loss', 'mae')
    model.compile(**compile_params)

    return model


def build_matrix_autoencoder(input_length, create_params, compile_params):
    """
    Builds several autoencoders each used for every feature vector in Matrix
    """

    inputs = [Input(shape=(create_params['input_shape'],)) for i in range(input_length)]

    models = [create_autoencoder(**create_params) for i in range(input_length)]
    
    outputs = [model(inputs[i]) for i, model in enumerate(models)]

    final_model = Model(inputs=inputs, outputs=outputs)
    compile_params['loss'] = compile_params.get('loss', 'mae')
    final_model.compile(**compile_params)

    return final_model
