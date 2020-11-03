import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from load_data import load_fashion_mnist_unscaled

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def create_model(show_summary=False):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation='relu'),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    if show_summary:
        model.summary()

    return model


def evaluate_history(history):
    print("History parameters: ", history.params)
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("./img/A1_history.png", dpi=300)
    plt.show()


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_fashion_mnist_unscaled()

    model = create_model(show_summary=True)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=0.01),
                  metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=3,
                        validation_data=(X_valid, y_valid))
    evaluate_history(history)

    # Evaluate
    model.evaluate(X_test, y_test)

    # Predict
    y_proba = model.predict(X_test[120:123])
    print(y_proba.round(2))
    y_class = model.predict_classes(X_test[120:123])
    print(y_class)


if __name__ == "__main__":
    main()
