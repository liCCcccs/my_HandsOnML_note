import tensorflow as tf
from tensorflow import keras
from load_data import load_fashion_mnist_unscaled
import numpy as np


def train_cnn_autoencoder(X_train, X_valid, n_epochs=10):
    """ Train an Convolutional Layers autoencoder """
    conv_encoder = keras.models.Sequential([
        keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
        keras.layers.Conv2D(16, kernel_size=3, padding="SAME", activation="selu"),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(32, kernel_size=3, padding="SAME", activation="selu"),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(64, kernel_size=3, padding="SAME", activation="selu"),
        keras.layers.MaxPool2D(pool_size=2)
    ])
    conv_decoder = keras.models.Sequential([
        keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="VALID", activation="selu",
                                     input_shape=[3, 3, 64]),
        keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="SAME", activation="selu"),
        keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="SAME", activation="sigmoid"),
        keras.layers.Reshape([28, 28])
    ])
    conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])

    conv_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.0))
    history = conv_ae.fit(X_train, X_train, epochs=n_epochs, validation_data=(X_valid, X_valid))

    return conv_encoder, conv_ae


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_fashion_mnist_unscaled()

    conv_encoder, conv_ae = train_cnn_autoencoder(X_train, X_valid, n_epochs=10)
    conv_ae.save("./saved_model/A5_cnn_autoencoder.h5")


if __name__ == "__main__":
    main()
