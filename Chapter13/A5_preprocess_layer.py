import tensorflow as tf
from tensorflow import keras
from load_data import load_california_housing_unscaled
from sklearn.preprocessing import StandardScaler
import numpy as np


class Standardization(keras.layers.Layer):
    """ A preprocess layer that can adapt to data and normalize it """
    def adapt(self, data_sample):
        self.means_ = np.mean(data_sample, axis=0, keepdims=True)
        self.stds_ = np.std(data_sample, axis=0, keepdims=True)

    def call(self, inputs):
        return (inputs - self.means_) / (self.stds_ + keras.backend.epsilon())


def create_model(input_shape, standardize_layer):
    model = keras.models.Sequential([
        standardize_layer,
        keras.layers.Dense(30, activation="relu", input_shape=input_shape),
        keras.layers.Dense(1)
    ])

    return model


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_california_housing_unscaled()
    standardize_layer = Standardization()
    standardize_layer.adapt(X_train[:500])

    model = create_model(X_train.shape[1:], standardize_layer)
    model.compile(optimizer="nadam", loss="mse", metrics=["mae"])
    model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))

    model.evaluate(X_test, y_test)


if __name__ == "__main__":
    main()
