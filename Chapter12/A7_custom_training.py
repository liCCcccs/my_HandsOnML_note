"""
Custom training loop
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import load_data


def random_batch(X, y, batch_size=32):
    """ Randomly generate a training batch from training data """
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]


def create_model(input_shape):
    l2_reg = keras.regularizers.l2(0.05)
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal",
                           kernel_regularizer=l2_reg),
        keras.layers.Dense(1, kernel_regularizer=l2_reg)
    ])

    return model


def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                         for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{}/{} - ".format(iteration, total) + metrics,
          end=end)


def custom_training_loop(X_train_scaled, X_valid_scaled, y_train, y_valid, model):
    n_epochs = 5
    batch_size = 32
    n_steps = len(X_train_scaled) // batch_size
    optimizer = keras.optimizers.Nadam(lr=0.01)
    loss_fn = keras.losses.mean_squared_error
    mean_loss = keras.metrics.Mean()
    metrics = [keras.metrics.MeanAbsoluteError()]

    for epoch in range(1, n_epochs + 1):
        print("Epoch {}/{}".format(epoch, n_epochs))
        for step in range(1, n_steps + 1):
            X_batch, y_batch = random_batch(X_train_scaled, y_train)
            with tf.GradientTape() as tape:
                y_pred = model(X_batch)
                main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                loss = tf.add_n([main_loss] + model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            for variable in model.variables:
                if variable.constraint is not None:
                    variable.assign(variable.constraint(variable))
            mean_loss(loss)
            for metric in metrics:
                metric(y_batch, y_pred)
            print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)
        print_status_bar(len(y_train), len(y_train), mean_loss, metrics)
        for metric in [mean_loss] + metrics:
            metric.reset_states()

    return model


def main():
    X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test = load_data.load_california_housing()

    model = create_model(X_train_scaled.shape[1:])

    trained_model = custom_training_loop(X_train_scaled, X_valid_scaled, y_train, y_valid, model)

    trained_model.save("./saved_model/A7_custom_training.h5")


if __name__ == "__main__":
    main()
