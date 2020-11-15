import tensorflow as tf
from tensorflow import keras
from load_data import download_bach_chorales, play_chords
import numpy as np
from pathlib import Path
import pandas as pd


def load_chorales(filepaths):
    return [pd.read_csv(filepath).values.tolist() for filepath in filepaths]


def collect_data(filepath):
    """ Extract data from jsb_chorales.tgz, convert to list """
    jsb_chorales_dir = Path(filepath).parent
    train_files = sorted(jsb_chorales_dir.glob("train/chorale_*.csv"))
    valid_files = sorted(jsb_chorales_dir.glob("valid/chorale_*.csv"))
    test_files = sorted(jsb_chorales_dir.glob("test/chorale_*.csv"))
    train_chorales = load_chorales(train_files)
    valid_chorales = load_chorales(valid_files)
    test_chorales = load_chorales(test_files)
    return train_chorales, valid_chorales, test_chorales


def create_target(batch):
    X = batch[:, :-1]
    Y = batch[:, 1:] # predict next note in each arpegio, at each step
    return X, Y


def preprocess(window):
    min_note = 36  # notes range from 36 to 81
    window = tf.where(window == 0, window, window - min_note + 1) # shift values
    return tf.reshape(window, [-1]) # convert to arpegio


def bach_dataset(chorales, batch_size=32, shuffle_buffer_size=None,
                 window_size=32, window_shift=16, cache=True):
    def batch_window(window):
        return window.batch(window_size + 1)

    def to_windows(chorale):
        dataset = tf.data.Dataset.from_tensor_slices(chorale)  # chorale: tensor: [length_of_music, 4], e.g. [192, 4]
        dataset = dataset.window(window_size + 1, window_shift, drop_remainder=True)  # [num_window, 33, 4]
        return dataset.flat_map(batch_window)

    chorales = tf.ragged.constant(chorales, ragged_rank=1)  # [num_data, length_of_music_not_same, 4], so ragged tensor
    dataset = tf.data.Dataset.from_tensor_slices(chorales)
    dataset = dataset.flat_map(to_windows)  # dataset shape [num_batch, 33, 4]
    dataset = dataset.map(preprocess)  # dataset shape [num_batch, 132]

    if cache:
        dataset = dataset.cache()
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(create_target)  # get [0], pred [1], get [0,1], pred [2], get [0,1,2], pred [3], ...
    return dataset.prefetch(1)


def create_model():
    n_embedding_dims = 5
    n_notes = 47
    model = keras.models.Sequential([
        keras.layers.Embedding(input_dim=n_notes, output_dim=n_embedding_dims,
                               input_shape=[None]),
        keras.layers.Conv1D(32, kernel_size=2, padding="causal", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(48, kernel_size=2, padding="causal", activation="relu", dilation_rate=2),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(64, kernel_size=2, padding="causal", activation="relu", dilation_rate=4),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(96, kernel_size=2, padding="causal", activation="relu", dilation_rate=8),
        keras.layers.BatchNormalization(),
        keras.layers.LSTM(256, return_sequences=True),
        keras.layers.Dense(n_notes, activation="softmax")
    ])
    model.summary()
    return model


def get_data():
    filepath = download_bach_chorales()  # filepath = "/home/mingruis/.keras/datasets/jsb_chorales/jsb_chorales.tgz"
    train_chorales, valid_chorales, test_chorales = collect_data(filepath)
    # train_chorales, valid_chorales, test_chorales: list [num_data, length_of_music_not_same, 4]

    # Export some example to listen to
    # for index in range(3):
    #     play_chords(train_chorales[index], filepath="./saved_audio/Bach_example" + str(index) + ".wav")

    train_set = bach_dataset(train_chorales, shuffle_buffer_size=1000)
    valid_set = bach_dataset(valid_chorales)
    test_set = bach_dataset(test_chorales)

    return train_set, valid_set, test_set


def train_save_model(train_set, valid_set):
    model = create_model()
    optimizer = keras.optimizers.Nadam(lr=1e-3)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["sparse_categorical_accuracy"])
    model.fit(train_set, epochs=20, validation_data=valid_set)

    model.save("./saved_model/Bach_generate.h5")


def generate_chorale(model, seed_chords, length):
    """
    Generate a pieces of music by predicting
    :param model:
    :param seed_chords: the first 8 chords, shape [2, 4]
    :param length: the length of music about to be generated
    :return: music
    """
    min_note = 36  # notes range from 36 to 81
    arpegio = preprocess(tf.constant(seed_chords, dtype=tf.int64))  # shape [32,]
    arpegio = tf.reshape(arpegio, [1, -1])  # shape [1, 32]
    for chord in range(length):
        for note in range(4):
            next_note = model.predict_classes(arpegio)[:1, -1:]  # after pred [1, 32], output [1, 1], take the last note
            arpegio = tf.concat([arpegio, next_note], axis=1)  # concat two tensor [1, 32+] and [1, 1] along axis 1
    arpegio = tf.where(arpegio == 0, arpegio, arpegio + min_note - 1)  # back to the original range
    return tf.reshape(arpegio, shape=[-1, 4])


def generate_save_music(seed=0, filename=None, temperature=None):
    """ Generate a piece of music and save it, seed can be any non-negative integer """
    filepath = download_bach_chorales()  # filepath = "/home/mingruis/.keras/datasets/jsb_chorales/jsb_chorales.tgz"
    _, _, test_chorales = collect_data(filepath)

    seed_chords = test_chorales[seed][:8]  # shape [8, 4]

    model = keras.models.load_model("./saved_model/Bach_generate.h5")
    if temperature is None:
        new_chorale = generate_chorale(model, seed_chords, 56)
    else:
        new_chorale = generate_chorale_v2(model, seed_chords, 56, temperature)

    play_chords(new_chorale, filepath=filename)


def generate_chorale_v2(model, seed_chords, length, temperature=1):
    """ Generate music more bravely, instead of keep the same note for long time """
    min_note = 36  # notes range from 36 to 81
    arpegio = preprocess(tf.constant(seed_chords, dtype=tf.int64))
    arpegio = tf.reshape(arpegio, [1, -1])
    for chord in range(length):
        for note in range(4):
            next_note_probas = model.predict(arpegio)[0, -1:]  # after pred [1, 32, 47], output [1, 47]
            rescaled_logits = tf.math.log(next_note_probas) / temperature  # [1, 47]
            next_note = tf.random.categorical(rescaled_logits, num_samples=1)  # [1, 1]
            arpegio = tf.concat([arpegio, next_note], axis=1)  # concat two tensor [1, 32+], [1, 1] along axis 1
    arpegio = tf.where(arpegio == 0, arpegio, arpegio + min_note - 1)
    return tf.reshape(arpegio, shape=[-1, 4])


def main():
    train_set, valid_set, test_set = get_data()

    # train_save_model(train_set, valid_set)

    generate_save_music(seed=1, filename="./saved_audio/generated_t08.wav", temperature=0.8)


if __name__ == "__main__":
    main()
