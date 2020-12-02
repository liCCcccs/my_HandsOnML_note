import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
from functools import partial
import matplotlib.pyplot as plt


def generate_3d_data(m, w1=0.1, w2=0.3, noise=0.1):
    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
    data = np.empty((m, 3))
    data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
    data[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
    data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * np.random.randn(m)
    return data


def plot_3d_scatter(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    plt.show()


def plot_encoded_2d_scatter(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 1])
    plt.show()


def create_auto_encoder():
    model = keras.models.Sequential([
        keras.layers.Dense(2, input_shape=[3]),
        keras.layers.Dense(3)
    ])
    return model


def main():
    X_train = generate_3d_data(60)
    X_train = X_train - X_train.mean(axis=0, keepdims=0)

    encoder = keras.models.Sequential([keras.layers.Dense(2, input_shape=[3])])
    decoder = keras.models.Sequential([keras.layers.Dense(3, input_shape=[2])])
    # connect two Sequential models, because one of them --- the encoder will be used separately
    model = keras.models.Sequential([encoder, decoder])

    model.compile(optimizer=keras.optimizers.SGD(lr=1.5), loss="mse")
    model.fit(X_train, X_train, epochs=20)

    x_encoded = encoder.predict(X_train)

    plot_encoded_2d_scatter(x_encoded)


if __name__ == "__main__":
    main()
