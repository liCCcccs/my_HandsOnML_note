"""
In this script, RNN only forecast one step ahead
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from load_data import generate_time_series, plot_series
import numpy as np


def separate_data(data, n_steps):
    X_train, y_train = data[:7000, :n_steps], data[:7000, -1]
    X_valid, y_valid = data[7000:9000, :n_steps], data[7000:9000, -1]
    X_test, y_test = data[9000:, :n_steps], data[9000:, -1]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def baseline_prediction(X_test, y_actual):
    """ Baseline prediction, use the last X as predicted value """
    y_pred = X_test[:, -1]
    mse_loss = tf.reduce_mean(keras.losses.mean_squared_error(y_actual, y_pred))
    print("MSE loss of baseline:", mse_loss)

    plot_series(X_test[0, :, 0], y_actual[0, :], y_pred[0])
    #plt.show()


def linear_prediction(X_train, y_train, X_valid, y_valid, n_steps):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[n_steps, 1]),
        keras.layers.Dense(1)
    ])
    model.summary()
    model.compile(optimizer="sgd", loss="mse")
    model.fit(X_train, y_train, epochs=20)

    model.evaluate(X_valid, y_valid)

    y_pred = model.predict(X_valid)
    plot_series(X_valid[0, :, 0], y_valid[0, :], y_pred[0])
    plt.show()


def simple_rnn_prediction(X_train, y_train, X_valid, y_valid):
    """ No need to know the input sequence length, since RNN can handle any length of input series """
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(1, input_shape=[None, 1])
    ])
    model.summary()
    model.compile(optimizer="sgd", loss="mse")
    model.fit(X_train, y_train, epochs=20)
    model.evaluate(X_valid, y_valid)

    y_pred = model.predict(X_valid)
    plot_series(X_valid[0, :, 0], y_valid[0, :], y_pred[0])
    plt.show()


def deep_rnn_prediction(X_train, y_train, X_valid, y_valid):
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20, return_sequences=True),
        keras.layers.SimpleRNN(1)
    ])
    model.summary()
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=20)
    model.evaluate(X_valid, y_valid)

    y_pred = model.predict(X_valid)
    plot_series(X_valid[0, :, 0], y_valid[0, :], y_pred[0])
    plt.show()


def deep_rnn_prediction_v2(X_train, y_train, X_valid, y_valid):
    """ make the second RNN layer return only the last output """
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(1)
    ])
    model.summary()
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=20)
    model.evaluate(X_valid, y_valid)
    model.save("./saved_model/A1_deepRNN_v2.h5")

    y_pred = model.predict(X_valid)
    plot_series(X_valid[0, :, 0], y_valid[0, :], y_pred[0])
    plt.show()


def main():
    n_steps = 50
    series = generate_time_series(num_series=10000, n_steps=n_steps + 1)  # shape [num_series, n_steps, 1]
    X_train, X_valid, X_test, y_train, y_valid, y_test = separate_data(series, n_steps)

    #baseline_prediction(X_valid, y_valid)

    #linear_prediction(X_train, y_train, X_valid, y_valid, n_steps)

    #simple_rnn_prediction(X_train, y_train, X_valid, y_valid, n_steps)

    #deep_rnn_prediction(X_train, y_train, X_valid, y_valid)

    deep_rnn_prediction_v2(X_train, y_train, X_valid, y_valid)


if __name__ == "__main__":
    main()
