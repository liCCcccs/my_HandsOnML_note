import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_preprocess_data():
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def create_model(input_shape):
    regularizer = tf.keras.regularizers.L2(l2=0.003)
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(300, activation="relu", kernel_regularizer=regularizer),
        keras.layers.Dense(200, activation="relu", kernel_regularizer=regularizer),
        keras.layers.Dense(100, activation="relu", kernel_regularizer=regularizer),
        keras.layers.Dense(10, activation='softmax')
    ])

    return model


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_preprocess_data()

    model = create_model(X_train.shape[1:])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
    acc = model.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()
