import tensorflow as tf
from tensorflow import keras
from load_data import load_fashion_mnist_unscaled
import numpy as np


def train_rnn_autoencoder(X_train, X_valid, n_epochs=10):
    """ Train an rnn autoencoder """
    recurrent_encoder = keras.models.Sequential([
        keras.layers.LSTM(100, return_sequences=True, input_shape=[None, 28]),
        keras.layers.LSTM(30)  # output a vector of length 30 that encode the information of input
    ])
    recurrent_decoder = keras.models.Sequential([
        keras.layers.RepeatVector(28, input_shape=[30]),
        keras.layers.LSTM(100, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(28, activation="sigmoid"))
    ])
    recurrent_ae = keras.models.Sequential([recurrent_encoder, recurrent_decoder])
    recurrent_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(0.1))
    history = recurrent_ae.fit(X_train, X_train, epochs=10, validation_data=(X_valid, X_valid))

    return recurrent_encoder, recurrent_ae


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_fashion_mnist_unscaled()

    recurrent_encoder, recurrent_ae = train_rnn_autoencoder(X_train, X_valid, n_epochs=10)
    recurrent_ae.save("./saved_model/A6_rnn_autoencoder.h5")


if __name__ == "__main__":
    main()
