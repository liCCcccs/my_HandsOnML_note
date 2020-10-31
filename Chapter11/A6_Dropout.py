"""
Dropout
Alpha dropout - use this when using self-normalization
MC dropout - predict multiple times with different dropout units, observe mean and var, but it will slow down prediction
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


def create_model_A(input_shape):
    """ Dropout, MC dropout also can use this model """
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dropout(rate=0.2),   # or keras.layers.AlphaDropout(rate=0.2),
        keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
        keras.layers.Dropout(rate=0.2),   # or keras.layers.AlphaDropout(rate=0.2),
        keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
        keras.layers.Dropout(rate=0.2),   # or keras.layers.AlphaDropout(rate=0.2),
        keras.layers.Dense(10, activation="softmax")
    ])

    return model


def create_model_B(input_shape):
    """ Self-normalization + Alpha dropout """
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.AlphaDropout(rate=0.2),
        keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.AlphaDropout(rate=0.2),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.AlphaDropout(rate=0.2),
        keras.layers.Dense(10, activation="softmax")
    ])

    return model


def predict_by_MCdropout(model, X_test_scalsed):
    y_probas = np.stack([model(X_test_scalsed, training=True)
                         for sample in range(100)])
    y_proba = y_probas.mean(axis=0)
    y_std = y_probas.std(axis=0)
    y_pred = np.argmax(y_proba, axis=1)

    return y_pred



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

    model = create_model_A(X_train.shape[1:])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["sparse_categorical_accuracy"])
    n_epochs = 2
    history = model.fit(X_train_scaled, y_train, epochs=1,
                        validation_data=(X_valid_scaled, y_valid))

    model.save("./saved_model/A6_model_with_dropout")

    # Evaluate accuracy without MC dropout
    model.evaluate(X_test_scaled, y_test)

    # Evaluate accuracy with MC dropout
    y_pred = predict_by_MCdropout(model, X_test_scaled)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(accuracy)


if __name__ == "__main__":
    main()
