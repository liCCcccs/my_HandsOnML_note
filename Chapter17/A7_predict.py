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


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_fashion_mnist_unscaled()

    auto_encoder = keras.models.load_model("./saved_model/A7_denoising_ae.h5")
    noise = keras.layers.GaussianNoise(stddev=0.2)
    predict_fig(auto_encoder, noise(X_test, training=True), 5)


if __name__ == "__main__":
    main()
