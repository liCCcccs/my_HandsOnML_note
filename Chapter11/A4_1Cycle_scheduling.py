import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

K = keras.backend


class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
        super().__init__()
        self.iterations = iterations   # total number of iterations: #epoch * #batch per epoch
        self.max_rate = max_rate   # max learning rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1   # define a very small portion of iteration
        self.half_iteration = (iterations - self.last_iterations) // 2   # start -> half_iter -> 2*half_iter -> end
        self.last_rate = last_rate or self.start_rate / 1000   # define a very small lr for last few iterations
        self.iteration = 0   # current iteration

    def _interpolate(self, iter1, iter2, rate1, rate2):
        """ linear mapping lr between iter1 and iter2 """
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)

    def on_batch_begin(self, batch, logs=None):
        if self.iteration < self.half_iteration:
            # first part: gradually increase lr
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            # second part: decrease lr
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            # last part: very small lr to finish the training
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
            rate = max(rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, rate)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)


def find_learning_rate(model, X, y, epochs=1, batch_size=32, min_rate=10**-5, max_rate=10):
    """ Run one epoch, exponentially decay lr per batch, then restore the model to initial.
        We need observe training history to determine
    """
    init_weights = model.get_weights()
    iterations = len(X) // batch_size * epochs
    factor = np.exp(np.log(max_rate / min_rate) / iterations)
    init_lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                        callbacks=[exp_lr])
    K.set_value(model.optimizer.lr, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses


def plot_lr_vs_loss(rates, losses):
    plt.plot(rates, losses)
    plt.gca().set_xscale('log')
    plt.hlines(min(losses), min(rates), max(rates))
    plt.axis([min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 2])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.show()


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


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_preprocess_data()
    # scale to zero mean and standard deviation 1
    pixel_means = X_train.mean(axis=0, keepdims=True)
    pixel_stds = X_train.std(axis=0, keepdims=True)
    X_train_scaled = (X_train - pixel_means) / pixel_stds
    X_valid_scaled = (X_valid - pixel_means) / pixel_stds
    X_test_scaled = (X_test - pixel_means) / pixel_stds

    # convert nan to 0
    X_train_scaled = np.nan_to_num(X_train_scaled)
    X_valid_scaled = np.nan_to_num(X_valid_scaled)
    X_test_scaled = np.nan_to_num(X_test_scaled)

    tf.random.set_seed(42)
    np.random.seed(42)

    model = create_model(X_train.shape[1:])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=1e-3),
                  metrics=["sparse_categorical_accuracy"])

    batch_size = 32
    which_phase = "1Cycle_scheduling"

    if which_phase == "Find_learning_rate":
        rates, losses = find_learning_rate(model, X_train_scaled, y_train, epochs=1, batch_size=batch_size)
        plot_lr_vs_loss(rates, losses)
    elif which_phase == "1Cycle_scheduling":
        n_epochs = 25
        onecycle = OneCycleScheduler(len(X_train[:400]) // batch_size * n_epochs, max_rate=0.05)
        history = model.fit(X_train_scaled[:400], y_train[:400], epochs=n_epochs, batch_size=batch_size,
                            validation_data=(X_valid_scaled, y_valid),
                            callbacks=[onecycle])

        plt.plot(history.epoch, history.history["lr"], "o-")
        plt.xlabel("Iteration")
        plt.ylabel("Learning Rate")
        plt.title("1Cycle Scheduling", fontsize=14)
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()

