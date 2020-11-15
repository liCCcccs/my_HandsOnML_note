"""
WaveNet, the stride is always 1
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from load_data import generate_time_series, plot_multiple_forecasts
import numpy as np


def last_time_step_mse(Y_true, Y_pred):
    """ For metrics, we only care about the last prediction """
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


def create_model():
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[None, 1]))
    for rate in (1, 2, 4, 8) * 2:
        model.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal",
                                      activation="relu", dilation_rate=rate))
    model.add(keras.layers.Conv1D(filters=10, kernel_size=1))
    return model


def main():
    n_steps = 50
    series = generate_time_series(10000, n_steps + 10)
    X_train = series[:7000, :n_steps]  # shape [7000, 50, 1]
    X_valid = series[7000:9000, :n_steps]  # shape 2000, 50, 1]
    X_test = series[9000:, :n_steps]  # shape [1000, 50, 1]
    Y = np.empty((10000, n_steps, 10))  # shape [10000, 50, 10]
    # Y: total number of data 10000, use 50 times in a single input of training, forecast 10 steps ahead
    for step_ahead in range(1, 10 + 1):
        Y[:, :, step_ahead - 1] = series[:, step_ahead:step_ahead + n_steps, 0]
    Y_train = Y[:7000]   # shape [7000, 50, 10]
    Y_valid = Y[7000:9000]   # shape [2000, 50, 10]
    Y_test = Y[9000:]   # shape [1000, 50, 10]

    model = create_model()

    model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.01), metrics=[last_time_step_mse])
    history = model.fit(X_train, Y_train, epochs=2,
                        validation_data=(X_valid, Y_valid))


if __name__ == "__main__":
    main()
