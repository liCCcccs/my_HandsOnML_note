import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_fashion_mnist_unscaled
import numpy as np


def create_model():
    encoder = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(30, activation='relu')
    ])
    decoder = keras.models.Sequential([
        keras.layers.Dense(100, activation='relu', input_shape=[30]),
        keras.layers.Dense(784, activation='sigmoid'),
        keras.layers.Reshape([28, 28])
    ])
    auto_encoder = keras.models.Sequential([encoder, decoder])
    return encoder, auto_encoder


def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_fashion_mnist_unscaled()

    encoder, auto_encoder = create_model()
    auto_encoder.compile(optimizer=keras.optimizers.SGD(lr=1.5), loss="binary_crossentropy", metrics=[rounded_accuracy])
    auto_encoder.fit(X_train, X_train, epochs=20, validation_data=(X_valid, X_valid))

    auto_encoder.save("./saved_model/A2_autoEncoder.h5")


if __name__ == "__main__":
    main()
