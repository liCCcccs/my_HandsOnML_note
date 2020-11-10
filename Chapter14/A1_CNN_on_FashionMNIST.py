import tensorflow as tf
from tensorflow import keras
from load_data import load_fashion_mnist_unscaled_normalized
from functools import partial
import numpy as np


def create_model():
    DefaultConv2D = partial(keras.layers.Conv2D,
                            kernel_size=3, activation='relu', padding="SAME")

    model = keras.models.Sequential([
        DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=128),
        DefaultConv2D(filters=128),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=256),
        DefaultConv2D(filters=256),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=10, activation='softmax'),
    ])

    return model


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_fashion_mnist_unscaled_normalized()
    X_train = X_train[..., np.newaxis]  # add a new axis, as the original image doesn't have a dimension on channel
    X_valid = X_valid[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    model = create_model()
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    #history = model.fit(X_train, y_train, epochs=3, validation_data=(X_valid, y_valid))

    score = model.evaluate(X_test, y_test)

    X_new = X_test[:10]  # pretend we have new images
    y_pred = model.predict(X_new)
    print(y_pred)




if __name__ == "__main__":
    main()
