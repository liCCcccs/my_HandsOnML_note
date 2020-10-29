"""
SELU activation function was proposed in this great paper by Günter Klambauer, Thomas Unterthiner and Andreas Mayr,
published in June 2017. During training, a neural network composed exclusively of a stack of dense layers using the
SELU activation function and LeCun initialization will self-normalize: the output of each layer will tend to preserve
the same mean and variance during training, which solves the vanishing/exploding gradients problem.

As a result, this activation function outperforms the other activation functions very significantly for such neural nets,
so you should really try it out. Unfortunately, the self-normalizing property of the SELU activation function is easily
broken: you cannot use ℓ1 or ℓ2 regularization, regular dropout, max-norm, skip connections or other non-sequential
topologies (so recurrent neural networks won't self-normalize).

However, in practice it works quite well with sequential CNNs. If you break self-normalization, SELU will not
necessarily outperform other activation functions.

"""
import tensorflow as tf
from tensorflow import keras
import numpy as np


def load_preprocess_data():
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def create_model(input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    model.add(keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"))
    # TODO: try 100 layers, when you have sick GPUs
    for layer in range(3):
        model.add(keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"))
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_preprocess_data()

    model = create_model(X_train.shape[1:])
    model.compile(optimizer=keras.optimizers.SGD(1e-3),
                  loss="sparse_categorical_crossentropy", metrics="sparse_categorical_accuracy")

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

    history = model.fit(X_train_scaled, y_train, epochs=5,
                        validation_data=(X_valid_scaled, y_valid))

    model.evaluate(X_test_scaled, y_test)


if __name__ == "__main__":
    main()
