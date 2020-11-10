"""
Define all function relevant to load data here
"""
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


def load_california_housing():
    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target.reshape(-1, 1), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test


def load_california_housing_unscaled(get_feature_name=False):
    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target.reshape(-1, 1), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42)

    if get_feature_name:
        return X_train, X_valid, X_test, y_train, y_valid, y_test, housing.feature_names

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def load_fashion_mnist_unscaled():
    """ should be unnormalized, already scaled """
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    X_valid, X_train = X_train_full[:5000] / 255, X_train_full[5000:] / 255
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def load_fashion_mnist_unscaled_normalized():
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
    y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

    X_mean = X_train.mean(axis=0, keepdims=True)
    X_std = X_train.std(axis=0, keepdims=True) + 1e-7
    X_train = (X_train - X_mean) / X_std
    X_valid = (X_valid - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def load_digit_mnist_unscaled():
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def generate_time_series(num_series, n_steps):
    """ generate a signal sequence for RNN """
    np.random.seed(42)
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, num_series, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(num_series, n_steps) - 0.5)   # + noise

    return series[..., np.newaxis].astype(np.float32)


def plot_series(series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$", n_steps=50):
    """ A1_basic_RNN.py, plot series data """
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bx", markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "ro")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1, 1])


def plot_multiple_forecasts(X, Y, Y_pred):
    """ A2_forecast_multiple_steps.py, plot series data """
    n_steps = X.shape[0]
    ahead = Y.shape[0]
    plot_series(X)
    plt.plot(np.arange(n_steps, n_steps + ahead), Y, "ro-", label="Actual")
    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred, "bx-", label="Forecast", markersize=10)
    plt.axis([0, n_steps + ahead, -1, 1])
    plt.legend(fontsize=14)

