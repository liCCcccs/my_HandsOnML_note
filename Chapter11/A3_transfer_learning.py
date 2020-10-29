import tensorflow as tf
from tensorflow import keras


def load_preprocess_data():
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def split_dataset(X, y):
    """ training set A: contain all classes except [y==5, y==6]
        training set B: contain only classes [y==5, y==6], only 200 examples
    """
    y_5_or_6 = (y == 5) | (y == 6)
    y_A = y[~y_5_or_6]
    y_A[y_A > 6] -= 2  # class indices 7, 8, 9 should be moved to 5, 6, 7
    y_B = y[y_5_or_6]
    y_B -= 5   # class indices 5,6 should be 0,1
    return (X[~y_5_or_6], y_A), (X[y_5_or_6], y_B)


def create_model(input_shape):
    model_A = keras.models.Sequential()
    model_A.add(keras.layers.Flatten(input_shape=[28, 28]))
    for n_hidden in (300, 100, 50, 50, 50):
        model_A.add(keras.layers.Dense(n_hidden, activation="selu"))
    model_A.add(keras.layers.Dense(8, activation="softmax"))

    return model_A


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_preprocess_data()
    (X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(X_train, y_train)
    (X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = split_dataset(X_valid, y_valid)
    (X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(X_test, y_test)
    X_train_B = X_train_B[:200]
    y_train_B = y_train_B[:200]

    model_A = create_model(X_train_A.shape[1:])
    model_A.compile(loss="sparse_categorical_crossentropy",
                    optimizer=keras.optimizers.SGD(lr=1e-3), metrics=["sparse_categorical_accuracy"])

    history = model_A.fit(X_train_A, y_train_A, epochs=20,
                          validation_data=(X_valid_A, y_valid_A))

    model_A.save("./saved_model/A3_raw_model.h5")


if __name__ == "__main__":
    main()




