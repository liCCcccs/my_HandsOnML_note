"""
In this script, RNN forecast multiple steps ahead
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from load_data import generate_time_series, plot_multiple_forecasts
import numpy as np


def separate_data(data, n_steps):
    X_train, y_train = data[:7000, :n_steps], data[:7000, -1]
    X_valid, y_valid = data[7000:9000, :n_steps], data[7000:9000, -1]
    X_test, y_test = data[9000:, :n_steps], data[9000:, -1]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def forecast_option1(X_valid, y_valid, n_steps):
    """ Using model trained for forecasting one step ahead, iteratively forecast multiple steps """
    model = keras.models.load_model("./saved_model/A1_deepRNN_v2.h5")
    batch_size = X_valid.shape[0]
    y_pred = np.zeros((batch_size, 10, 1), dtype=np.float32)

    num_steps = 10  # want to forecast 10 steps ahead
    for step in range(num_steps):
        y_next = model.predict(tf.concat([X_valid[:, step:n_steps], y_pred[:, :step]], axis=1))
        y_pred[:, step, :] = y_next[:, 0]

    mse = tf.reduce_mean(keras.losses.mean_squared_error(y_valid, y_pred))
    print("MSE of forecast_option1:", mse)

    plot_multiple_forecasts(X_valid[0, :, 0], y_valid[0, :], y_pred[0])
    plt.show()


def naive_forecast(X_valid, y_valid, n_steps):
    batch_size = X_valid.shape[0]
    num_steps = 10  # want to forecast 10 steps ahead
    y_pred = np.zeros((batch_size, num_steps, 1))
    for i in range(batch_size):
        y_pred[i, :, :] = np.full((num_steps, 1), X_valid[i, -1, 0])

    mse = tf.reduce_mean(keras.losses.mean_squared_error(y_valid, y_pred))
    print("MSE of naive_forecast:", mse)

    plot_multiple_forecasts(X_valid[0, :, 0], y_valid[0, :], y_pred[0])
    plt.show()


def linear_forecast(X_train, Y_train, X_valid, y_valid, n_steps):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[50, 1]),
        keras.layers.Dense(10)
    ])

    model.compile(loss="mse", optimizer="adam")
    history = model.fit(X_train, Y_train, epochs=20,
                        validation_data=(X_valid, y_valid))
    mse = model.evaluate(X_valid, y_valid)
    print("MSE of naive_forecast:", mse)

    y_pred = model.predict(X_valid[:1, :, :])
    plot_multiple_forecasts(X_valid[0, :, 0], y_valid[0, :], y_pred[0])
    plt.show()


def deep_rnn_forecast(X_train, y_train, X_valid, y_valid):
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(10)
    ])

    model.compile(loss="mse", optimizer="adam")
    history = model.fit(X_train, y_train, epochs=20,
                        validation_data=(X_valid, y_valid))

    y_pred = model.predict(X_valid[:1, :, :])
    plot_multiple_forecasts(X_valid[0, :, 0], y_valid[0, :], y_pred[0])
    plt.show()


def main():
    n_steps = 50
    series = generate_time_series(10000, n_steps + 10)
    X_train, y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
    X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
    X_test, y_test = series[9000:, :n_steps], series[9000:, -10:, 0]
    X_new, Y_new = series[:, :n_steps], series[:, n_steps:]  # X_new [num_series, n_steps, 50], y_new [num_series, n_steps, 1]

    #X_train, X_valid, X_test, y_train, y_valid, y_test = separate_data(series, n_steps)

    #forecast_option1(X_new, Y_new, n_steps)

    #naive_forecast( X_new, Y_new, n_steps)

    #linear_forecast(X_train, y_train, X_valid, y_valid, n_steps)

    deep_rnn_forecast(X_train, y_train, X_valid, y_valid)


if __name__ == "__main__":
    main()
