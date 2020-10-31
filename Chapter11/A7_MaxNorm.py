"""
When using MC dropout together with batch normalization, need some special treatment
Can't directly set training=true when predicting
"""


import tensorflow as tf
from tensorflow import keras
from functools import partial
import numpy as np


def load_preprocess_data():
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def create_model(input_shape):
    """ model with Max norm """
    MaxNormDense = partial(keras.layers.Dense, activation="selu", kernel_initializer="lecun_normal",
                       kernel_constraint=keras.constraints.max_norm(1.))

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        MaxNormDense(300),
        MaxNormDense(100),
        MaxNormDense(10, activation="softmax")
    ])

    return model


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_preprocess_data()
    # scale to zero mean and standard deviation 1
    pixel_means = X_train.mean(axis=0, keepdims=True)
    pixel_stds = X_train.std(axis=0, keepdims=True)
    X_train_scaled = (X_train - pixel_means) / pixel_stds
    X_valid_scaled = (X_valid - pixel_means) / pixel_stds
    X_test_scaled = (X_test - pixel_means) / pixel_stds

    # convert nan to 0
    X_train_scaled = np.nan_to_num(X_train_scaled)
    X_valid_scaled = np.nan_to_num(X_valid_scaled)
    X_test_scaled = np.nan_to_num(X_test_scaled)

    model = create_model(X_train.shape[1:])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["sparse_categorical_accuracy"])
    history = model.fit(X_train_scaled, y_train, epochs=2,
                        validation_data=(X_valid_scaled, y_valid))


if __name__ == "__main__":
    main()

