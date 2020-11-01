"""
Load the model with a custom loss class
The hyperparameters are also loaded when loading the model
"""

import tensorflow as tf
from tensorflow import keras
import load_data


class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    def get_config(self):
        base_config = super().get_config()  # base_config is a dictionary of all hyperparameters
        return {**base_config, "threshold": self.threshold}


def main():
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_data.load_california_housing()

    model = keras.models.load_model("./saved_model/A1_model_custom_loss_class.h5",
                                    custom_objects={"HuberLoss": HuberLoss})

    # Check if the hyperparameter "threshold" is loaded
    print("threshold", model.loss.threshold)

    model.fit(X_train_scaled, y_train, epochs=2,
              validation_data=(X_valid_scaled, y_valid))


if __name__ == "__main__":
    main()
