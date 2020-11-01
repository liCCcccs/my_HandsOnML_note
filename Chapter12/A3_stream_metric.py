"""
Create a stream metric
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import load_data


class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs) # handles base args (e.g., dtype)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = threshold * tf.abs(error) - threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn


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
    model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=[HuberMetric(2.0)])
    model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32), epochs=2)

    print(model.metrics[-1].threshold)  # this hyper-parameter can be saved with saved_model


if __name__ == "__main__":
    main()
