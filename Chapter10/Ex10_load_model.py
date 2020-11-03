import tensorflow as tf
from tensorflow import keras
from load_data import load_digit_mnist_unscaled


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_digit_mnist_unscaled()

    model = keras.models.load_model("./saved_model/Ex10_best.h5")

    acc = model.evaluate(X_test, y_test)

    tf.print(acc)


if __name__ == "__main__":
    main()
