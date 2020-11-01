"""
Load the model with a custom loss function
"""

import tensorflow as tf
from tensorflow import keras

import load_data


def huber_fn(y_true, y_pred):
    """ Need to define the custom loss function here """
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)


def main():
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_data.load_california_housing()

    model = keras.models.load_model("./saved_model/A1_model_custom_loss_func.h5",
                                    custom_objects={"huber_fn": huber_fn})

    model.fit(X_train_scaled, y_train, epochs=2,
              validation_data=(X_valid_scaled, y_valid))


if __name__ == "__main__":
    main()
