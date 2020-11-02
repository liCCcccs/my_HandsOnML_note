"""
Load model with custom layer
"""

import tensorflow as tf
from tensorflow import keras
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


def main():
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_data.load_california_housing()

    model = keras.models.load_model("./saved_model/A4_custom_layer.h5", custom_objects={"MyDense": MyDense})
    model.fit(X_train_scaled, y_train, epochs=2,
              validation_data=(X_valid_scaled, y_valid))

    model.evaluate(X_test_scaled, y_test)


if __name__ == "__main__":
    main()
