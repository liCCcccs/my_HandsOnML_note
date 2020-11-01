"""
Define Huber loss in a function and save the model
Also, custom Activation Functions, Initializers, Regularizers, and Constraints can also be done this way
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_california_housing


def huber_fn(y_true, y_pred):
    """ Args are arrays """
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)


def plot_huber_fn():
    plt.figure(figsize=(8, 3.5))
    z = np.linspace(-4, 4, 200)
    plt.plot(z, huber_fn(0, z), "b-", linewidth=2, label="huber($z$)")
    plt.plot(z, z ** 2 / 2, "b:", linewidth=1, label=r"$\frac{1}{2}z^2$")
    plt.plot([-1, -1], [0, huber_fn(0., -1.)], "r--")
    plt.plot([1, 1], [0, huber_fn(0., 1.)], "r--")
    plt.gca().axhline(y=0, color='k')
    plt.gca().axvline(x=0, color='k')
    plt.axis([-4, 4, 0, 4])
    plt.grid(True)
    plt.xlabel("$z$")
    plt.legend(fontsize=14)
    plt.title("Huber loss", fontsize=14)
    plt.show()


def create_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                           input_shape=input_shape),
        keras.layers.Dense(1),
    ])

    return model


def main():
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_california_housing()

    model = create_model(X_train_scaled.shape[1:])
    model.compile(loss=huber_fn, optimizer="nadam", metrics=["mae"])

    model.fit(X_train_scaled, y_train, epochs=10,
              validation_data=(X_valid_scaled, y_valid))

    model.save("./saved_model/A1_model_custom_loss_func.h5")


if __name__ == "__main__":
    main()
