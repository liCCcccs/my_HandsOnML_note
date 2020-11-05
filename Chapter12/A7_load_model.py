"""
Load model custom trained model
"""

import tensorflow as tf
from tensorflow import keras
import load_data


def main():
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_data.load_california_housing()

    model = keras.models.load_model("./saved_model/A7_custom_training.h5")
    model.compile(metrics=["mae"])
    model.evaluate(X_test_scaled, y_test)


if __name__ == "__main__":
    main()
