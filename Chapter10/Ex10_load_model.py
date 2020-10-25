import tensorflow as tf
from tensorflow import keras


def load_preprocess_data():
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_preprocess_data()

    model = keras.models.load_model("./saved_model/Ex10_best.h5")

    acc = model.evaluate(X_test, y_test)

    tf.print(acc)









if __name__ == "__main__":
    main()
