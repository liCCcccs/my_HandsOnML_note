import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
from functools import partial

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

INPUT_CHARS = "".join(sorted(set("".join(MONTHS)))) + "01234567890, "
TARGET_CHARS = "".join(sorted(set("1234567890-")))


def id_to_char(ids, chars=INPUT_CHARS):
    return "".join([chars[i - 1] for i in ids if i != 0])


def char_to_id(s, chars=INPUT_CHARS):
    return [chars.index(char) for char in s]


def preprocess_input(s):
    s_id = char_to_id(s, chars=INPUT_CHARS)
    s_id = np.array(s_id) + 1
    while len(s_id) != 18:
        s_id = np.append(s_id, 0)  # pad with zero to keep all input sequence the same length as the longest sequence
    # the model needs input shape [batch_size, sequence_length], in this case we make batch_size = 1
    return np.expand_dims(s_id, axis=0)

def test_model_1():
    model = keras.models.load_model("./saved_model/Ex9_simple_encoder_decoder.h5")

    X_test = preprocess_input("September 17, 2009")
    print(X_test)
    y_test = model.predict_classes(X_test)
    y_decoded = id_to_char(y_test[0], chars=TARGET_CHARS)
    print(y_decoded)

    X_test = preprocess_input("September 30, 2020")
    y_test = model.predict_classes(X_test)
    y_decoded = id_to_char(y_test[0], chars=TARGET_CHARS)
    print(y_decoded)

    X_test = preprocess_input("February 2, 1998")
    y_test = model.predict_classes(X_test)
    y_decoded = id_to_char(y_test[0], chars=TARGET_CHARS)
    print(y_decoded)


def test_model_2():
    sos_id = len(TARGET_CHARS)+1
    model = keras.models.load_model("./saved_model/Ex9_encoder_decoder_v2.h5")
    max_output_length = 10
    X = preprocess_input("July 14, 1789")  # X = [[1,2,3,4,5,...]] for single input
    Y_pred = tf.fill(dims=(len(X), 1), value=sos_id)  # shape [len(X), 1], value is 12
    for index in range(max_output_length):
        pad_size = max_output_length - Y_pred.shape[1]  # from 9, 8, 7, ...
        X_decoder = tf.pad(Y_pred, [[0, 0], [0, pad_size]])  # e.g. [[12]] -> [[12,0,0,0,0,0,...]]
        Y_probas_next = model.predict([X, X_decoder])[:, index:index + 1]
        Y_pred_next = tf.argmax(Y_probas_next, axis=-1, output_type=tf.int32)  # predict on char ahead
        Y_pred = tf.concat([Y_pred, Y_pred_next], axis=1)
    y_output = id_to_char(Y_pred[0, 1:], chars=TARGET_CHARS)
    print(y_output)


def test_model_2_tfa():
    sos_id = len(TARGET_CHARS)+1
    basicDecoder = tfa.seq2seq.basic_decoder.BasicDecoder
    model = keras.models.load_model("./saved_model/Ex9_encoder_decoder_v2.h5", custom_objects={"BasicDecoder": basicDecoder})
    max_output_length = 10
    X = preprocess_input("July 14, 1789")  # X = [[1,2,3,4,5,...]] for single input
    Y_pred = tf.fill(dims=(len(X), 1), value=sos_id)  # shape [len(X), 1], value is 12
    for index in range(max_output_length):
        pad_size = max_output_length - Y_pred.shape[1]  # from 9, 8, 7, ...
        X_decoder = tf.pad(Y_pred, [[0, 0], [0, pad_size]])  # e.g. [[12]] -> [[12,0,0,0,0,0,...]]
        Y_probas_next = model.predict([X, X_decoder])[:, index:index + 1]
        Y_pred_next = tf.argmax(Y_probas_next, axis=-1, output_type=tf.int32)  # predict on char ahead
        Y_pred = tf.concat([Y_pred, Y_pred_next], axis=1)
    y_output = id_to_char(Y_pred[0, 1:], chars=TARGET_CHARS)
    print(y_output)


def test_model_2_tfa_v2():
    """ This is another way to predict, much faster than iteratively predicting, instead,
        it predict just once for each sequence """
    basicDecoder = tfa.seq2seq.basic_decoder.BasicDecoder
    model = keras.models.load_model("./saved_model/Ex9_encoder_decoder_v2.h5",
                                    custom_objects={"BasicDecoder": basicDecoder})
    model.summary()
    decoder_embedding_layer = model.get_layer("embedding_1")
    output_layer = model.get_layer("dense")
    units = 128
    decoder_cell = keras.layers.LSTMCell(units)
    max_output_length = 10
    encoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
    sos_id = len(TARGET_CHARS) + 1
    encoder_outputs, state_h, state_c = model.get_layer("lstm").output
    encoder_state = [state_h, state_c]

    print(decoder_embedding_layer)

    inference_sampler = tfa.seq2seq.sampler.GreedyEmbeddingSampler(
        embedding_fn=decoder_embedding_layer)
    inference_decoder = tfa.seq2seq.basic_decoder.BasicDecoder(
        decoder_cell, inference_sampler, output_layer=output_layer,
        maximum_iterations=max_output_length)
    batch_size = tf.shape(encoder_inputs)[:1]
    start_tokens = tf.fill(dims=batch_size, value=sos_id)
    final_outputs, final_state, final_sequence_lengths = inference_decoder(
        start_tokens,
        initial_state=encoder_state,
        start_tokens=start_tokens,
        end_token=0)
    Y_proba = keras.layers.Activation("softmax")(final_outputs.rnn_output)
    inference_model = keras.models.Model(inputs=[encoder_inputs],
                                         outputs=[Y_proba])  # TODO: this is hard

    #X = prepare_date_strs_padded(date_strs)
    #Y_pred = inference_model.predict(X)
    #return ids_to_date_strs(Y_pred)


def main():
    test_model_2_tfa_v2()


if __name__ == "__main__":
    main()
