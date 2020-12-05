import tensorflow as tf
from tensorflow import keras
from load_data import load_fashion_mnist_unscaled
import numpy as np



def train_autoencoder(n_neurons, X_train, X_valid, loss, optimizer,
                      n_epochs=10, output_activation=None, metrics=None):
    """ Train an one-layer-encoder + one-layer-decoder autoencoder """
    n_inputs = X_train.shape[-1]   # X_train shape [batch_size, length]
    encoder = keras.models.Sequential([
        keras.layers.Dense(units=n_neurons, activation="selu", input_shape=[n_inputs])
    ])
    decoder = keras.models.Sequential([
        keras.layers.Dense(units=n_inputs, activation=output_activation)
    ])
    model = keras.models.Sequential([encoder, decoder])
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(X_train, X_train, epochs=n_epochs, validation_data=(X_valid, X_valid))

    return encoder, decoder, encoder.predict(X_train), encoder.predict(X_valid)


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_fashion_mnist_unscaled()

    K = keras.backend
    X_train_flatten = K.batch_flatten(X_train)
    X_valid_flatten = K.batch_flatten(X_valid)

    enc1, dec1, X_train_2, X_valid_2 = train_autoencoder(n_neurons=100, X_train=X_train_flatten,
                                                         X_valid=X_valid_flatten, loss="binary_crossentropy",
                                                         optimizer="nadam", output_activation="sigmoid", n_epochs=10)
    enc2, dec2, _, _ = train_autoencoder(n_neurons=100, X_train=X_train_2, X_valid=X_valid_2,
                                         loss="mse", optimizer="nadam", output_activation="selu", n_epochs=10)

    auto_encoder = keras.models.Sequential([enc1, enc2, dec2, dec1])
    auto_encoder.save("./saved_model/A4_autoencoder.h5")



if __name__ == "__main__":
    main()
