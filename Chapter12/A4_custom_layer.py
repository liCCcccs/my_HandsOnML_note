"""
1. Layer without weights
    Lambda layer, e.g. exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))
2. Layer with weights
3. Layer with multiple inputs and outputs
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import load_data


class MyDense(keras.layers.Layer):
    """ Define a dense layer manually """

    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)  # activation is e.g. "relu", "selu"

    def build(self, batch_input_shape):
        """ This method is called when first use this layer """
        self.kernel = self.add_weight(
            name="kernel", shape=[batch_input_shape[-1], self.units],
            initializer="glorot_normal")
        self.bias = self.add_weight(
            name="bias", shape=[self.units], initializer="zeros")
        super().build(batch_input_shape)  # must be at the end

    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)

    def compute_output_shape(self, batch_input_shape):
        """ Return the output shape of this layer.
            We can generally omit this method, as tf.keras automatically infers the output shape,
            except when the layer is dynamic
        """
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": keras.activations.serialize(self.activation)}


class MyMultiLayer(keras.layers.Layer):
    """ This layer has two inputs and two outputs,
        so it can only use the Functional and Subclassing APIs, not Sequential APIs
    """

    def call(self, X):
        X1, X2 = X
        return X1 + X2, X1 * X2

    def compute_output_shape(self, batch_input_shape):
        batch_input_shape1, batch_input_shape2 = batch_input_shape
        return [batch_input_shape1, batch_input_shape2]  # output shape is the same as input shape


def create_model_1(input_shape):
    """ create a model with the custom layer """
    model = keras.models.Sequential([
        MyDense(units=100, activation="relu", input_shape=input_shape),
        MyDense(units=1),
    ])

    return model


def main():
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_data.load_california_housing()

    model = create_model_1(X_train_scaled.shape[1:])
    model.compile(loss="mse", optimizer="nadam")
    model.fit(X_train_scaled, y_train, epochs=6,
              validation_data=(X_valid_scaled, y_valid))

    model.evaluate(X_test_scaled, y_test)

    model.save("./saved_model/A4_custom_layer.h5")


if __name__ == "__main__":
    main()
