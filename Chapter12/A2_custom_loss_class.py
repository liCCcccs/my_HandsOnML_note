"""
Define Huber loss in a class and save the model
In this way, we can save hyper-parameters of the loss when saving the model

Also, custom Activation Functions, Initializers, Regularizers, and Constraints with hyper-parameters needed
to be saved with model can also be done this way
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


def create_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                           input_shape=input_shape),
        keras.layers.Dense(1),
    ])

    return model


def main():
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_data.load_california_housing()

    model = create_model(X_train_scaled.shape[1:])
    model.compile(loss=HuberLoss(2.0), optimizer="nadam", metrics=["mae"])

    model.fit(X_train_scaled, y_train, epochs=10,
              validation_data=(X_valid_scaled, y_valid))

    model.save("./saved_model/A1_model_custom_loss_class.h5")


if __name__ == "__main__":
    main()
