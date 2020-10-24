import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_preprocess_data():
    housing = fetch_california_housing()

    X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # fit and transform
    X_valid = scaler.transform(X_valid)  # transform
    X_test = scaler.transform(X_test)  # transform

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def create_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=[8]),
        keras.layers.Dense(30, activation="relu"),
        keras.layers.Dense(1)
    ])

    return model


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_preprocess_data()

    model = create_model(input_shape=8)
    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
    mse_test = model.evaluate(X_test, y_test)

    y_pred1 = model.predict(X_test[:3])
    print("y_pred1", y_pred1)

    model.save("./saved_model/A5_model.h5")

    model = keras.models.load_model("./saved_model/A5_model.h5")
    y_pred2 = model.predict(X_test[:3])
    print("y_pred2", y_pred2)


if __name__ == "__main__":
    main()
