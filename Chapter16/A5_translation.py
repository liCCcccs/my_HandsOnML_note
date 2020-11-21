import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np


def create_model(vocab_size, embed_size):
    encoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
    decoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
    sequence_lengths = keras.layers.Input(shape=[], dtype=np.int32)

    embeddings = keras.layers.Embedding(vocab_size, embed_size)
    encoder_embeddings = embeddings(encoder_inputs)
    decoder_embeddings = embeddings(decoder_inputs)

    encoder = keras.layers.LSTM(512, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_embeddings)
    encoder_state = [state_h, state_c]

    sampler = tfa.seq2seq.sampler.TrainingSampler()

    decoder_cell = keras.layers.LSTMCell(512)
    output_layer = keras.layers.Dense(vocab_size)
    decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell, sampler,
                                                     output_layer=output_layer)
    final_outputs, final_state, final_sequence_lengths = decoder(
        decoder_embeddings, initial_state=encoder_state,
        sequence_length=sequence_lengths)
    Y_proba = tf.nn.softmax(final_outputs.rnn_output)

    model = keras.models.Model(
        inputs=[encoder_inputs, decoder_inputs, sequence_lengths],
        outputs=[Y_proba])

    return model


def main():
    model = create_model(vocab_size=100, embed_size=50)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    # pretend this is the training set
    X = np.random.randint(100, size=10 * 1000).reshape(1000, 10)
    Y = np.random.randint(100, size=15 * 1000).reshape(1000, 15)
    X_decoder = np.c_[np.zeros((1000, 1)), Y[:, :-1]]
    seq_lengths = np.full([1000], 15)  # the length of decoded sequence is 15

    history = model.fit([X, X_decoder, seq_lengths], Y, epochs=1)


if __name__ == "__main__":
    main()
