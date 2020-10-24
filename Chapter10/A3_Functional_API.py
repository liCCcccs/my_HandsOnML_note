import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_preprocess_data():
    housing = fetch_california_housing()

    X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=41)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=41)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # fit and transform
    X_valid = scaler.transform(X_valid)  # transform
    X_test = scaler.transform(X_test)  # transform

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def preprocess_data_model2(X_train, X_valid, X_test):
    inputA_train, inputB_train = X_train[:, :5], X_train[:, 2:]
    inputA_valid, inputB_valid = X_valid[:, :5], X_valid[:, 2:]
    inputA_test, inputB_test = X_test[:, :5], X_test[:, 2:]

    return inputA_train, inputB_train, inputA_valid, inputB_valid, inputA_test, inputB_test


def create_model(input_shape, show_summary=False):
    input_ = keras.layers.Input(shape=input_shape)
    hidden1 = keras.layers.Dense(30, activation='relu')(input_)
    hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
    concat = keras.layers.concatenate([input_, hidden2])
    output = keras.layers.Dense(1)(concat)
    model = keras.models.Model(inputs=[input_], outputs=[output])

    if show_summary:
        model.summary()

    return model


def create_model_wide_deep(show_summary=False):
    inputA = keras.layers.Input(shape=[5])
    inputB = keras.layers.Input(shape=[6])
    hidden1 = keras.layers.Dense(30, activation='relu')(inputA)
    hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
    concat = keras.layers.concatenate(inputs=[inputB, hidden2])
    output = keras.layers.Dense(1)(concat)
    model = keras.models.Model(inputs=[inputA, inputB], outputs=output)

    if show_summary:
        model.summary()

    return model


def create_model_aux_ouput(show_summary=False):
    inputA = keras.layers.Input(shape=[5])
    inputB = keras.layers.Input(shape=[6])
    hidden1 = keras.layers.Dense(30, activation='relu')(inputA)
    hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
    concat = keras.layers.concatenate(inputs=[inputB, hidden2])
    output = keras.layers.Dense(1, name="main_output")(concat)
    aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
    model = keras.models.Model(inputs=[inputA, inputB], outputs=[output, aux_output])

    if show_summary:
        model.summary()

    return model


def main_model1():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_preprocess_data()

    model = create_model(X_train.shape[1:], show_summary=True)

    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

    mse_test = model.evaluate(X_test, y_test)
    print("mse_test: ", mse_test)

    X_new = X_test[:3]
    y_pred = model.predict(X_new)
    print("y_pred", y_pred)


def main_model2():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_preprocess_data()
    inputA_train, inputB_train, inputA_valid, inputB_valid, inputA_test, inputB_test = preprocess_data_model2(X_train, X_valid, X_test)

    model = create_model_wide_deep(show_summary=True)

    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
    history = model.fit((inputA_train, inputB_train), y_train, epochs=20,
                        validation_data=((inputA_valid, inputB_valid), y_valid))

    mse_test = model.evaluate((inputA_test, inputB_test), y_test)
    print("mse_test: ", mse_test)

    X_newA, X_newB = inputA_test[:3], inputB_test[:3],
    y_pred = model.predict((X_newA, X_newB))
    print("y_pred", y_pred)


def main_model3():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_preprocess_data()
    inputA_train, inputB_train, inputA_valid, inputB_valid, inputA_test, inputB_test = preprocess_data_model2(X_train, X_valid, X_test)

    model = create_model_aux_ouput(show_summary=True)

    model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(lr=1e-3))
    history = model.fit((inputA_train, inputB_train), y_train, epochs=20,
                        validation_data=((inputA_valid, inputB_valid), (y_valid, y_valid)))

    mse_test = model.evaluate((inputA_test, inputB_test), (y_test, y_test))
    print("mse_test: ", mse_test)

    X_newA, X_newB = inputA_test[:3], inputB_test[:3],
    y_pred_main, y_pred_aux = model.predict((X_newA, X_newB))
    print("y_pred_main", y_pred_main)
    print("y_pred_aux", y_pred_aux)


if __name__ == "__main__":
    # main_model1()
    # main_model2()
    main_model3()
