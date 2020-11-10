import tensorflow as tf
from tensorflow import keras
from load_data import load_fashion_mnist_unscaled_normalized
from functools import partial
import numpy as np


class ResidueModule(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


def create_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, input_shape=[28, 28, 1]))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="SAME"))
    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidueModule(filters, strides=strides))
        prev_filters = filters
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1000, activation="softmax"))

    model.summary()

    return model


def main():
    create_model()


if __name__ == "__main__":
    main()
