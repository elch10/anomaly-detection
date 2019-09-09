from keras import Sequential
from keras.layers import Dense, Input, Dropout
from keras.regularizers import l2
from keras.models import Model
import keras
import numpy as np

def build_autoencoder(input_shape, layers_size, reg_strength=0.01, 
                      dropout_fraction=0.1, **compile_attrs):
    """
    Builds autoencoder model with hidden layers of size `layers_size`
    """
    model = Sequential()

    model.add(Dense(layers_size[0],
            input_shape=(input_shape,),
            kernel_regularizer=l2(reg_strength)))

    for size in layers_size[1:]:
        model.add(Dense(
            size,
            kernel_regularizer=l2(reg_strength),
        ))

    compile_attrs['loss'] = compile_attrs.get('loss', 'mae')
    model.compile(**compile_attrs)
    return model


def build_matrix_autoencoder(input_length, input_shape, layers_size, 
                             reg_strength=0.01, dropout_fraction=0.1, **compile_attrs):
    
    def create_autoencoder(layers_size, reg_strength, dropout_fraction):
        model = Sequential()
        model.add(Dropout(dropout_fraction))
        for layer_size in layers_size[:-1]:
            model.add(Dense(layer_size, activation='tanh', kernel_regularizer=l2(reg_strength)))
        model.add(Dense(layers_size[-1], kernel_regularizer=l2(reg_strength)))
        return model

    inputs = [Input(shape=(input_shape,)) for i in range(input_length)]

    models = [create_autoencoder(layers_size, reg_strength, dropout_fraction) for i in range(input_length)]
    
    outputs = [model(inputs[i]) for i, model in enumerate(models)]

    final_model = Model(inputs=inputs, outputs=outputs)
    compile_attrs['loss'] = compile_attrs.get('loss', 'mae')
    final_model.compile(**compile_attrs)

    return final_model
