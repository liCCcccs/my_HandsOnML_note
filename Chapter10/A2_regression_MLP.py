import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from load_data import load_california_housing


def create_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation='relu', input_shape=input_shape),
        keras.layers.Dense(1)
    ])

    return model


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_california_housing()

    model = create_model(X_train.shape[1:])
    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

    mse_test = model.evaluate(X_test, y_test)
    print("mse_test: ", mse_test)

    X_new = X_test[:3]
    y_pred = model.predict(X_new)
    print("y_pred", y_pred)


if __name__ == "__main__":
    main()








