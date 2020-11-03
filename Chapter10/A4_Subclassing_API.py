import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from load_data import load_california_housing


class WideAndDeepModel(keras.models.Model):
    def __init__(self, units, activation='relu', **kwargs):
        super().__init__(self, **kwargs)
        self.layer_hidden1 = keras.layers.Dense(units=units, activation=activation)
        self.layer_hidden2 = keras.layers.Dense(units=units, activation=activation)
        self.layer_main_out = keras.layers.Dense(units=1)
        self.layer_aux_out = keras.layers.Dense(units=1)

    def call(self, input):
        inputA, inputB = input
        hidden1 = self.layer_hidden1(inputA)
        hidden2 = self.layer_hidden2(hidden1)
        concat = keras.layers.concatenate([hidden2, inputA])
        main_out = self.layer_main_out(concat)
        aux_out = self.layer_aux_out(hidden2)

        return main_out, aux_out


def preprocess_data_model2(X_train, X_valid, X_test):
    inputA_train, inputB_train = X_train[:, :5], X_train[:, 2:]
    inputA_valid, inputB_valid = X_valid[:, :5], X_valid[:, 2:]
    inputA_test, inputB_test = X_test[:, :5], X_test[:, 2:]

    return inputA_train, inputB_train, inputA_valid, inputB_valid, inputA_test, inputB_test


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_california_housing()
    inputA_train, inputB_train, inputA_valid, inputB_valid, inputA_test, inputB_test = preprocess_data_model2(X_train, X_valid, X_test)

    model = WideAndDeepModel(units=30)
    model.compile(optimizer='adam', loss='mse', loss_weights=[0.9, 0.1])

    history = model.fit((inputA_train, inputB_train), (y_train, y_train), epochs=2,
                        validation_data=((inputA_valid, inputB_valid), (y_valid, y_valid)))

    total_loss, main_loss, aux_loss = model.evaluate((inputA_test, inputB_test), (y_test, y_test))
    y_pred_main, y_pred_aux = model.predict((inputA_test[:3], inputB_test[:3]))

    print("y_pred_main", y_pred_main, "y_pred_aux", y_pred_aux)


if __name__ == "__main__":
    main()



