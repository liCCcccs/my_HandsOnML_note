"""
When using MC dropout together with batch normalization, need some special treatment
Can't directly set training=true when predicting
"""


import tensorflow as tf
from tensorflow import keras
from functools import partial
import numpy as np


class MCDropoutLayer(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)  # whenever call it, during training or predicting, always training=True


def load_preprocess_data():
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def create_model_dropout(model_raw):
    mc_model = keras.models.Sequential([
        MCDropoutLayer(layer.rate) if isinstance(layer, keras.layers.Dropout) else layer  # layer.rate: dropout rate
        for layer in model_raw.layers
    ])

    mc_model.summary()

    return mc_model


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

    model_raw = keras.models.load_model("./saved_model/A6_model_with_dropout")

    mc_model = create_model_dropout(model_raw)

    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    mc_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["sparse_categorical_accuracy"])
    mc_model.set_weights(model_raw.get_weights())

    y_proba = np.round(np.mean([mc_model.predict(X_test_scaled[:1]) for sample in range(10)], axis=0), 2)
    print(y_proba)


if __name__ == "__main__":
    main()
