import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_fashion_mnist_unscaled
import numpy as np
from sklearn.manifold import TSNE


def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")


def predict_fig(auto_encoder, X_test, n_images):
    fig_pred = auto_encoder.predict(X_test[:n_images, :, :])
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(X_test[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(fig_pred[image_index])
    plt.show()


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


def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_fashion_mnist_unscaled()

    encoder = keras.models.load_model("./saved_model/A3_encoder.h5",
                                      custom_objects={"rounded_accuracy": rounded_accuracy, "DenseTranspose": DenseTranspose})
    auto_encoder = keras.models.load_model("./saved_model/A3_autoEncoder.h5",
                                           custom_objects={"rounded_accuracy": rounded_accuracy,
                                                           "DenseTranspose": DenseTranspose})  # TODO: can't load model this way
    predict_fig(auto_encoder, X_test, 5)

    X_valid_compressed = encoder.predict(X_valid)
    tsne = TSNE()
    X_valid_2D = tsne.fit_transform(X_valid_compressed)
    X_valid_2D = (X_valid_2D - X_valid_2D.min()) / (X_valid_2D.max() - X_valid_2D.min())

    plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap="tab10")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()







