"""
Custom model: build a residual module
"""

import tensorflow as tf
from tensorflow import keras
import load_data


class ReconstructingRegressor(keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(30, activation="selu",
                                          kernel_initializer="lecun_normal")
                       for _ in range(5)]
        self.out = keras.layers.Dense(output_dim)
        self.reconstruct = keras.layers.Dense(8)  # if not using build, we have to clarify the input dimension here

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)  # in order for this line to work, can't have build() method in the class
        return self.out(Z)


def main():
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_data.load_california_housing()

    model = ReconstructingRegressor(1)
    model.compile(loss="mse", optimizer="nadam")
    history = model.fit(X_train_scaled, y_train, epochs=2)


if __name__ == "__main__":
    main()
