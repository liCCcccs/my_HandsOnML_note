"""
Aim: solve gradient vanish/explosion
Two ways:
    1. Batch normalization after a layer
    2. Batch normalization in a layer, before activation function. (remember to cancel previous bias)

Solve gradient explosion, can also do:
Gradient Clipping
optimizer = keras.optimizers.SGD(clipvalue=1.0)
optimizer = keras.optimizers.SGD(clipnorm=1.0)
"""


import tensorflow as tf
from tensorflow import keras


def load_preprocess_data():
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def create_model1(input_shape):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(10, activation="softmax")
    ])

    return model


def create_model2(input_shape):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(300, use_bias=False),  # remember to not use bias
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dense(100, use_bias=False),  # remember to not use bias
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    return model


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_preprocess_data()

    model = create_model2(X_train.shape[1:])
    model.compile(optimizer=keras.optimizers.SGD(1e-3),
                  loss="sparse_categorical_crossentropy", metrics="sparse_categorical_accuracy")

    history = model.fit(X_train, y_train, epochs=5,
                        validation_data=(X_valid, y_valid))

    model.evaluate(X_test, y_test)


if __name__ == "__main__":
    main()


