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


def get_decoder_input(y):
    """ Generate decoder input, e.g. `<sos> I would like a glass of orange` for
        y=`I would like a glass of orange juice` """
    sos_id = len(TARGET_CHARS) + 1
    sos_column = tf.fill(dims=[len(y), 1], value=sos_id)
    return tf.concat([sos_column, y[:, :-1]], axis=1)


def create_model():
    """ Encoder-decoder architecture """
    encoder_inputs = keras.layers.Input(shape=[None], dtype=tf.int32)
    encoder_embedded = keras.layers.Embedding(input_dim=len(INPUT_CHARS) + 1,  # plus <pad>
                                              output_dim=300)(encoder_inputs)
    _, hidden_h, hidden_c = keras.layers.LSTM(128, return_state=True)(encoder_embedded)

    encoder_state = [hidden_h, hidden_c]
    decoder_inputs = keras.layers.Input(shape=[None], dtype=tf.int32)
    decoder_embedded = keras.layers.Embedding(input_dim=len(TARGET_CHARS) + 2,  # plus <pad> and <sos>
                                              output_dim=300)(decoder_inputs)
    decoder_lstm_outputs = keras.layers.LSTM(128, return_sequences=True)(decoder_embedded, initial_state=encoder_state)
    decoder_outputs = keras.layers.Dense(units=len(TARGET_CHARS)+1, activation="softmax")(decoder_lstm_outputs)

    model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[decoder_outputs])

    return model


def create_model_tfa():
    """ Using TF-Addons to build the same model as in function create_model() """
    encoder_embedding_size = 32
    decoder_embedding_size = 32
    units = 128

    encoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
    decoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
    sequence_lengths = keras.layers.Input(shape=[], dtype=np.int32)

    encoder_embeddings = keras.layers.Embedding(
        len(INPUT_CHARS) + 1, encoder_embedding_size)(encoder_inputs)

    decoder_embedding_layer = keras.layers.Embedding(
        len(INPUT_CHARS) + 2, decoder_embedding_size)
    decoder_embeddings = decoder_embedding_layer(decoder_inputs)

    encoder = keras.layers.LSTM(units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_embeddings)
    encoder_state = [state_h, state_c]

    sampler = tfa.seq2seq.sampler.TrainingSampler()

    decoder_cell = keras.layers.LSTMCell(units)
    output_layer = keras.layers.Dense(len(TARGET_CHARS) + 1)

    decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell,
                                                     sampler,
                                                     output_layer=output_layer)
    final_outputs, final_state, final_sequence_lengths = decoder(
        decoder_embeddings,
        initial_state=encoder_state)
    Y_proba = keras.layers.Activation("softmax")(final_outputs.rnn_output)

    model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs],
                               outputs=[Y_proba])

    return model


def main():
    data_size = 10000
    X_train, y_train = create_dataset(data_size)
    X_valid, y_valid = create_dataset(2000)
    X_test, y_test = create_dataset(2000)

    X_train_decoder = get_decoder_input(y_train)
    X_valid_decoder = get_decoder_input(y_valid)
    X_test_decoder = get_decoder_input(y_test)

    model = create_model_tfa()
    model.compile(optimizer='nadam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    history = model.fit([X_train, X_train_decoder], y_train, epochs=10,
                        validation_data=([X_valid, X_valid_decoder], y_valid))

    model.save("./saved_model/Ex9_encoder_decoder_v2_tfa.h5")


if __name__ == "__main__":
    main()
