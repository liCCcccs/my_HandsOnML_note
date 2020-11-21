import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow_hub as hub


def prepare_dataset(batch_size):
    datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
    train_size = info.splits["train"].num_examples
    train_set = datasets["train"].repeat().batch(batch_size)

    return train_set.prefetch(1), train_size


def create_model():
    """ This embedding accepts sentence as input, so no need to preprocess """
    model = keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
                       dtype=tf.string, input_shape=[], output_shape=[50]),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    return model


def main():
    batch_size = 32
    train_set, train_size = prepare_dataset(batch_size=batch_size)
    model = create_model()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(train_set, steps_per_epoch=train_size // batch_size, epochs=1)


if __name__ == "__main__":
    main()
