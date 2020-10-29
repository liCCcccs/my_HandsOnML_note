"""
Learning Rate Scheduling
1. Power Scheduling
    lr = lr0 / (1 + steps / s)**c
    $ optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)
2. Exponential Scheduling
    lr = lr0 * 0.1**(epoch / s)
    USE callback function
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

K = keras.backend


class ExponentialDecay(keras.callbacks.Callback):
    def __init__(self, s):
        super().__init__()
        self.s = s

    def on_batch_begin(self, batch, logs=None):
        lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, lr * 0.1 ** (1/self.s))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


class PiecewiseConstant(keras.callbacks.Callback):
    """ e.g. a = [5, 15], b = [1e-2, 5e-3, 1e-3] """
    def __init__(self, a, b):
        super().__init__()
        self.a = np.array([0] + a)
        self.b = np.array(b)

    def on_epoch_begin(self, epoch, logs=None):
        K.set_value(self.model.optimizer.lr, self.b[np.argmax(self.a > epoch) - 1])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


def load_preprocess_data():
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def create_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])

    return model


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_preprocess_data()

    model = create_model(X_train.shape[1:])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Nadam(lr=1e-2), metrics=["sparse_categorical_accuracy"])

    n_epoch = 20
    which_scheduling = "Performance_scheduling"

    if which_scheduling == "Exponential_per_epoch":
        exponential_decay_fn = exponential_decay(lr0=0.01, s=20)
        lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)  # Automatically carry lr0 to next epoch
    elif which_scheduling == "Exponential_per_batch":
        s = n_epoch * len(X_train) // 32  # this is the total number of batches.
                                        # Such that the lr should be 10 times smaller exactly when training finishes
        lr_scheduler = ExponentialDecay(s=s)
    elif which_scheduling == "Piecewise_constant":
        lr_scheduler = PiecewiseConstant(a=[5, 15], b=[1e-2, 5e-3, 1e-3])
    elif which_scheduling == "Performance_scheduling":
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

    # 200 is just to make it run faster
    history = model.fit(X_train[:200], y_train[:200], epochs=n_epoch,
                        validation_data=(X_valid, y_valid), callbacks=[lr_scheduler])

    model.evaluate(X_test, y_test)

    plt.plot(history.epoch, history.history["lr"], "o-")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title(which_scheduling, fontsize=14)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
