import utils

from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dropout, LSTM, ReLU, Activation, Dense
from keras.activations import relu


def build_baseline(shape_1, shape_2, n_vocab):
    model = Sequential()
    model.add(LSTM(256, input_shape=(shape_1, shape_2),
                   return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='kullback_leibler_divergence', optimizer='rmsprop')
    return model
