import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
from functools import partial


MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

INPUT_CHARS = "".join(sorted(set("".join(MONTHS)))) + "01234567890, "
TARGET_CHARS = "".join(sorted(set("1234567890-")))


def prepare_data(data_size=100):
    """ Generate a list of datetime, and corresponding standard representation """
    year_list = np.random.randint(low=1000, high=9999, size=[data_size])
    month_list = np.random.randint(low=0, high=11, size=[data_size])
    date_list = np.random.randint(low=1, high=31, size=[data_size])
    input_set = [MONTHS[month_list[i]] + ' ' + str(date_list[i]) + ', ' + str(year_list[i]) for i in range(data_size)]
    target_set = [str(year_list[i]) + '-' + str(month_list[i] + 1) + '-' + str(date_list[i]) for i in range(data_size)]
    return input_set, target_set


def char_to_id(s, chars=INPUT_CHARS):
    return [chars.index(char) for char in s]


def id_to_char(ids, chars=INPUT_CHARS):
    return "".join([chars[i-1] for i in ids.numpy()])


def create_dataset(data_size=100):
    """ Generate Input and Target sequence as tensors """
    input_set, target_set = prepare_data(data_size)
    data_X = [char_to_id(s, chars=INPUT_CHARS) for s in input_set]
    data_y = [char_to_id(s, chars=TARGET_CHARS) for s in target_set]
    data_X = tf.ragged.constant(data_X, ragged_rank=1)
    data_y = tf.ragged.constant(data_y, ragged_rank=1)
    return (data_X+1).to_tensor(), (data_y+1).to_tensor()  # leave index=0 to be <pad>


def create_model(embedding_size, max_output_length):
    """ The encoder first read the whole input sequence, the LSTM doesn't set return_sequence=True, so it only outputs
        the last hidden state. Then the last hidden state is repeated as many times as the decoder requires, as the
        decoder's input sequence.
    """
    encoder = keras.models.Sequential([
        # input_dim is the size of vocabulary, i.e. len([<pad>] + INPUT_CHARS)
        keras.layers.Embedding(input_dim=len(INPUT_CHARS) + 1, output_dim=embedding_size, input_shape=[None]),
        keras.layers.LSTM(128)
    ])

    decoder = keras.models.Sequential([
        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.Dense(len(TARGET_CHARS) + 1, activation="softmax")
    ])

    model = keras.models.Sequential([
        encoder,
        keras.layers.RepeatVector(max_output_length),
        decoder
    ])

    return model


def main():
    data_size = 10000
    X_train, y_train = create_dataset(data_size)
    X_valid, y_valid = create_dataset(2000)
    X_test, y_test = create_dataset(2000)

    print(X_train[0])
    print(y_train[0])
    print(id_to_char(X_train[0], INPUT_CHARS))
    print(id_to_char(y_train[0], TARGET_CHARS))

    model = create_model(embedding_size=32, max_output_length=y_train.shape[1])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

    model.save("./saved_model/Ex9_simple_encoder_decoder.h5")


if __name__ == "__main__":
    main()
