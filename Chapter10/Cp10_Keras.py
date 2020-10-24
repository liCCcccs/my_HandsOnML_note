import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


Class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

class PrintValTrainRatioCallback(keras.callbacks.Callback):
    """ custom callback here """
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))


def check_version():
    print("TensorFlow version: ", tf.__version__)  # '2.3.0'
    print("Keras version: ", keras.__version__)  # '2.4.0'

def load_fashion_MNIST():
    fashion_mnist = keras.datasets.fashion_mnist
    return fashion_mnist.load_data()

def check_property(X_train_full):
    print("X_train shape: ", X_train_full.shape)
    print("X_train dtype: ", X_train_full.dtype)

def normalise_divide(X_train_full, y_train_full, X_test):
    """ Normalize input X, and divide some data to validation set """
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.0
    return X_train, X_valid, y_train, y_valid, X_test

def create_NN():
    model = keras.models.Sequential()  # composed of a single stack of layers connected sequentially
    model.add(keras.layers.Flatten(input_shape=[28, 28], name="myInput"))  # convert each input image into a 1D array
    #model.add(keras.layers.InputLayer(input_shape=[28, 28], name="myInput"))  # equivalent as above
    model.add(keras.layers.Dense(300, activation="relu", name="myDense1"))
    model.add(keras.layers.Dense(100, activation="relu", name="myDense2"))
    model.add(keras.layers.Dense(10, activation="softmax", name="myDense3"))

    # Or equivalently
    #model = keras.models.Sequential([
    #    keras.layers.Flatten(input_shape=[28, 28]),
    #    keras.layers.Dense(300, activation="relu"),
    #    keras.layers.Dense(100, activation="relu"),
    #    keras.layers.Dense(10, activation="softmax")
    #])

    return model

def access_layer_from_model(model):
    """  two ways to get a layer from model, and access their weights"""
    layer_myDense1 = model.get_layer("myDense1")
    print(layer_myDense1.name)
    layer_myDense2 = model.layers[2]
    print(layer_myDense2.name)

    weights1, biases1 = layer_myDense1.get_weights()
    print("weights of layer myDense1: ", weights1)
    weights2, biases2 = layer_myDense2.get_weights()
    print("weights of layer myDense2: ", weights2)

    #layer_myDense2.set_weights((weights1, biases1))  # only when weights and biases of two layers have the same shape

def plot_history(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
    plt.show()

def do_prediction(X_test, y_test, model):
    """ predict the first 3 samples in the test set """
    X_new = X_test[:3]
    y_proba = model.predict(X_new)
    print("the probability of belonging to each class: ", y_proba.round(2))
    y_pred = model.predict_classes(X_new)
    print("the index of the predicted class: ", y_pred)
    print("the predicted class: ", np.array(Class_names)[y_pred])

    y_new = y_test[:3]
    print("true class: ", y_new)

def init_tensorbard():
    root_logdir = os.path.join(os.curdir, "my_logs")
    def get_run_logdir():
        import time
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)
    run_logdir = get_run_logdir()
    return run_logdir

def classification_MLP():
    """Run regression MLP on fashion MNIST data set, including saving model callback"""
    # pre-process data
    check_version()
    (X_train_full, y_train_full), (X_test, y_test) = load_fashion_MNIST()
    check_property(X_train_full)
    X_train, X_valid, y_train, y_valid, X_test = normalise_divide(X_train_full, y_train_full, X_test)

    # build model and access model
    model = create_NN()
    print(model.summary())
    #access_layer_from_model(model)

    # compile model
    model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(lr=0.01), metrics=["accuracy"])

    # define checkpoint to save model
    checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)

    # early stopping callback
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    # you can custom callback
    my_callback = PrintValTrainRatioCallback()

    # tensorboard callback
    run_logdir = init_tensorbard()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    # train
    # use `validation_split` to let keras split validation data (last few samples)
    # if you want split validation data manually, use `validation_data`
    history = model.fit(X_train, y_train, epochs=3, validation_data=(X_valid, y_valid), verbose=2, callbacks=[tensorboard_cb])

    # evaluate on test set
    model.evaluate(X_test, y_test)

    do_prediction(X_test, y_test, model)

    plot_history(history)



if __name__ == "__main__":

    classification_MLP()











"""end"""
