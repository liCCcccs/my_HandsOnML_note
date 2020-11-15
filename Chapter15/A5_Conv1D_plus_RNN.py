"""
This explains why drop the first time steps in Y
1D conv layer with kernel size 4, stride 2, VALID padding:

              |-----2-----|     |-----5---...------|     |-----23----|
        |-----1-----|     |-----4-----|   ...      |-----22----|
  |-----0----|      |-----3-----|     |---...|-----21----|
X: 0  1  2  3  4  5  6  7  8  9  10 11 12 ... 42 43 44 45 46 47 48 49
Y: 1  2  3  4  5  6  7  8  9  10 11 12 13 ... 43 44 45 46 47 48 49 50
  /10 11 12 13 14 15 16 17 18 19 20 21 22 ... 52 53 54 55 56 57 58 59, which means Y: 1/10, 2/11, 3/12, ...

Output:

X:     0/3   2/5   4/7   6/9   8/11 10/13 .../43 42/45 44/47 46/49
Y:     4/13  6/15  8/17 10/19 12/21 14/23 .../53 46/55 48/57 50/59  # the 1st Y is 4/13, so the first 3 steps is dropped
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
    model = keras.models.Sequential([
        keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid",
                            input_shape=[None, 1]),
        keras.layers.GRU(20, return_sequences=True),
        keras.layers.GRU(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(10))
    ])
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
    history = model.fit(X_train, Y_train[:, 3::2], epochs=2,
                        validation_data=(X_valid, Y_valid[:, 3::2]))


if __name__ == "__main__":
    main()
