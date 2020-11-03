"""
Custom model: build a residual module
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


class ResidualRegressor(keras.models.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.same_dimension = 30
        self.hidden1 = keras.layers.Dense(self.same_dimension , activation="elu",
                                          kernel_initializer="he_normal")   # `n_neurons` should has the same dimension as `inputs`
        self.block1 = ResidualBlock(2, self.same_dimension )  # `n_neurons` should has the same dimension as `inputs`
        self.block2 = ResidualBlock(2, self.same_dimension )  # `n_neurons` should has the same dimension as `inputs`
        self.out = keras.layers.Dense(output_dim)

    def call(self, inputs):
        Z = self.hidden1(inputs)
        for _ in range(1 + 3):
            Z = self.block1(Z)
        Z = self.block2(Z)
        return self.out(Z)

    #def get_config(self):
    #    """ implement this if want to save model """
    #    base_config = super().get_config()
    #    return {**base_config, "output_dim": self.output_dim}


def create_model():
    """ We can also create the model using Sequential API """
    block1 = ResidualBlock(2, 30)
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal"),
        block1, block1, block1, block1,
        ResidualBlock(2, 30),
        keras.layers.Dense(1)
    ])

    return model


def main():
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_data.load_california_housing()

    model = create_model()  # using Sequential model, have to define get_config() in ResidualBlock if want to save model
    # model = ResidualRegressor(1)  # or using Subclassing model, need to save as .ckpt
    model.compile(loss="mse", optimizer="nadam")

    history = model.fit(X_train_scaled, y_train, epochs=2)
    model.evaluate(X_test_scaled, y_test)

    model.save("./saved_model/A5_custom_model.h5")


if __name__ == "__main__":
    main()
