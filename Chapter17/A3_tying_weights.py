import tensorflow as tf
from tensorflow import keras
from load_data import load_fashion_mnist_unscaled
import numpy as np


class DenseTranspose(keras.layers.Layer):
    """ This layer's weight is the transpose of the weight in `dense_layer`, it bias is independent """
    def __init__(self, dense_layer, activation="relu", **kwargs):
        self.dense = dense_layer
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias",
                                      shape=[self.dense.input_shape[-1]],
                                      initializer="zeros")

    def call(self, inputs):
        # weights[0] is weight, weights[1] is bias
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True) + self.biases
        return self.activation(z)

    def get_config(self):
        base_config = super().get_config()  # base_config is a dictionary of all hyperparameters
        return {**base_config, "dense": self.dense, "activation": self.activation}


def create_model():
    dense1 = keras.layers.Dense(100, activation='relu', name="my_dense1")
    dense2 = keras.layers.Dense(30, activation='relu', name="my_dense2")
    encoder = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        dense1,
        dense2
    ])
    decoder = keras.models.Sequential([
        DenseTranspose(encoder.get_layer(name="my_dense2"), activation='relu', input_shape=[30]),
        DenseTranspose(encoder.get_layer(name="my_dense1"), activation='sigmoid'),
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
    auto_encoder.fit(X_train, X_train, epochs=1, validation_data=(X_valid, X_valid))

    auto_encoder.save("./saved_model/A3_autoEncoder.h5")
    encoder.save("./saved_model/A3_encoder.h5")


if __name__ == "__main__":
    main()
