import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_fashion_mnist_unscaled
import numpy as np


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


def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_fashion_mnist_unscaled()

    auto_encoder = keras.models.load_model("./saved_model/A2_autoEncoder.h5", custom_objects={"rounded_accuracy": rounded_accuracy})
    predict_fig(auto_encoder, X_test, 5)


if __name__ == "__main__":
    main()







