"""
In this script, we train an RNN that predicts the next 10 steps at each time step. That is,
instead of just forecasting time steps 50 to 59 based on time steps 0 to 49, it will forecast
time steps 1 to 10 at time step 0, then time steps 2 to 11 at time step 1, and so on, and finally
it will forecast time steps 50 to 59 at the last time step.
It can predict multiple steps ahead
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from load_data import generate_time_series, plot_multiple_forecasts
import numpy as np


class LNSimpleRNNCell(keras.layers.Layer):
    """ Layer Normalization normalize across features, Batch Normalization normalize across batches """
    def __init__(self, units, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.state_size = units
        self.output_size = units
        self.simple_rnn_cell = keras.layers.SimpleRNNCell(units,
                                                          activation=None)
        self.layer_norm = keras.layers.LayerNormalization()
        self.activation = keras.activations.get(activation)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        return [tf.zeros([batch_size, self.state_size], dtype=dtype)]

    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states)
        norm_outputs = self.activation(self.layer_norm(outputs))
        return norm_outputs, [norm_outputs]  # the first is output, the second is hidden state, this is special case
                # since in simple RNN, hidden state is equal to output


def create_model():
    """ Model input [batch, 50, 1], output [batch, 50, 10] """
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(10))
    ])
    return model


def create_model_batch_normalization():
    """ Batch normalization has little benefit to RNN """
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.BatchNormalization(),
        keras.layers.SimpleRNN(20, return_sequences=True),
        keras.layers.BatchNormalization(),
        keras.layers.TimeDistributed(keras.layers.Dense(10))
    ])
    return model


def create_model_layer_normalization():
    model = keras.models.Sequential([
        keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True,
                         input_shape=[None, 1]),
        keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(10))
    ])
    return model


def last_time_step_mse(Y_true, Y_pred):
    """ For metrics, we only care about the last prediction """
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


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

    #model = create_model()
    #model = create_model_batch_normalization()
    model = create_model_layer_normalization()

    model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.01), metrics=[last_time_step_mse])
    history = model.fit(X_train, Y_train, epochs=20,
                        validation_data=(X_valid, Y_valid))

    series = generate_time_series(1, 50 + 10)
    X_new, Y_new = series[:, :50, :], series[:, 50:, :]  # X_new [1, 50, 1], Y_new [1, 10, 1]
    Y_pred = model.predict(X_new)[:, -1, :][..., np.newaxis]  # predicted [1,50,10] -> [1,10] -> [1,10,1]

    plot_multiple_forecasts(X_new[0], Y_new[0], Y_pred[0])
    plt.show()


if __name__ == "__main__":
    main()
