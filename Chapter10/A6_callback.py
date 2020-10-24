import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))


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


def get_callbacks(callback_name):
    all_callbacks = []
    if "save_best_only" in callback_name:
        checkpoint_cb = keras.callbacks.ModelCheckpoint("./saved_model/A6_best.h5", save_best_only=True)
        all_callbacks.append(checkpoint_cb)
    if "EarlyStopping" in callback_name:
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        all_callbacks.append(early_stopping_cb)
    if "custom_callback" in callback_name:
        val_train_ratio_cb = PrintValTrainRatioCallback()
        all_callbacks.append(val_train_ratio_cb)

    return all_callbacks


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_preprocess_data()

    model = create_model(input_shape=8)
    model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

    all_callbacks = get_callbacks(callback_name=["save_best_only", "EarlyStopping", "custom_callback"])
    history = model.fit(X_train, y_train, epochs=30,
                        validation_data=(X_valid, y_valid),
                        callbacks=all_callbacks)


if __name__ == "__main__":
    main()
