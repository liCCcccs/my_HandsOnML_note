"""
Load model with custom model
"""

import tensorflow as tf
from tensorflow import keras
import load_data


class ResidualBlock(keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(n_neurons, activation="elu",
                                          kernel_initializer="he_normal")
                       for _ in range(n_layers)]
        self.n_layers = n_layers
        self.n_neurons = n_neurons

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z  # to keep this correct, make sure `n_neurons` and `inputs` have the same dimentsion

    def get_config(self):
        """ implement this if want to save model """
        base_config = super().get_config()
        return {**base_config, "n_layers": self.n_layers, "n_neurons": self.n_neurons}


def main():
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_data.load_california_housing()

    model = keras.models.load_model("./saved_model/A5_custom_model.h5", custom_objects={"ResidualBlock": ResidualBlock})

    history = model.fit(X_train_scaled, y_train, epochs=2)
    model.evaluate(X_test_scaled, y_test)


if __name__ == "__main__":
    main()
