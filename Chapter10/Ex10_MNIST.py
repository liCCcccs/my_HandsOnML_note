import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from load_data import load_digit_mnist_unscaled

K = keras.backend


class ExponentialLearningRate(keras.callbacks.Callback):
    """ increase learning rate every batch, default batch size == 32 """
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)


def create_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation='softmax')
    ])

    return model


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_digit_mnist_unscaled()

    model = create_model(X_train.shape[1:])
    # IMPORTANT: use "sparse_categorical_accuracy" instead of "accuracy"
    model.compile(optimizer=keras.optimizers.SGD(lr=5e-1), loss="sparse_categorical_crossentropy", metrics="sparse_categorical_accuracy")

    # callbacks
    # expon_lr = ExponentialLearningRate(factor=0.999)  # do this first to check learning rate
    checkpoint_cb = keras.callbacks.ModelCheckpoint("./saved_model/Ex10_best.h5", save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid), callbacks=[checkpoint_cb])

    model.evaluate(X_test, y_test)


if __name__ == "__main__":
    main()
