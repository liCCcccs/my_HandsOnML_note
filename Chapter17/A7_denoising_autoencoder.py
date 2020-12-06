import tensorflow as tf
from tensorflow import keras
from load_data import load_fashion_mnist_unscaled
import numpy as np


def train_denoising_autoencoder(X_train, X_valid, n_epochs=10):
    """ Train an de-noising autoencoder """
    denoise_encoder = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.GaussianNoise(stddev=0.2),
        keras.layers.Dense(units=100, activation="selu"),
        keras.layers.Dense(units=30, activation="selu")
    ])
    denoise_decoder = keras.models.Sequential([
        keras.layers.Dense(units=30, activation="selu"),
        keras.layers.Dense(units=28*28, activation="sigmoid"),
        keras.layers.Reshape([28, 28])
    ])
    denoise_ae = keras.models.Sequential([denoise_encoder, denoise_decoder])
    denoise_ae.compile(optimizer="nadam", loss="binary_crossentropy")
    denoise_ae.fit(X_train, X_train, epochs=n_epochs, validation_data=(X_valid, X_valid))

    return denoise_ae


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_fashion_mnist_unscaled()

    recurrent_ae = train_denoising_autoencoder(X_train, X_valid, n_epochs=10)
    recurrent_ae.save("./saved_model/A7_denoising_ae.h5")


if __name__ == "__main__":
    main()
