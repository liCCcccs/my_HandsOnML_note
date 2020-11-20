import tensorflow as tf
from tensorflow import keras
import numpy as np
from load_data import load_shakespeare


def prepare_dataset(shakespeare_text, batch_size=32, n_steps=100):
    """
    Prepare the training set, each sample of shape X [batch_size, n_steps, num_category], Y [batch_size, n_steps]
    :param shakespeare_text:
    :param batch_size:
    :param n_steps:
    :return:
    """
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(shakespeare_text)

    max_id = len(tokenizer.word_index)  # number of distinct characters
    dataset_size = tokenizer.document_count  # total number of characters

    [encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1  # convert character to integer

    train_size = dataset_size * 90 // 100
    window_length = n_steps + 1  # target = input shifted 1 character ahead

    def method_1():
        """ Only support batch size = 1 for stateful RNN """
        dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
        dataset = dataset.repeat().window(window_length, shift=n_steps, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(window_length))
        dataset = dataset.batch(1)
        dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))  # a window: [batch_size, window_length]
        dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))

        return dataset

    def method_2():
        """ Can have batch size > 1 for stateful RNN """
        encoded_parts = np.array_split(encoded[:train_size], batch_size)  # split it into n batches
        datasets = []  # a list of n datasets, e.g [{1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15}, ...]
        for encoded_part in encoded_parts:
            dataset = tf.data.Dataset.from_tensor_slices(encoded_part)
            dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
            dataset = dataset.flat_map(lambda window: window.batch(window_length))
            datasets.append(dataset)
        dataset = tf.data.Dataset.zip(tuple(datasets))  # e.g. {{1,6,11,...}, {2,7,12,...}, {3,8,13,...}, ...}
        dataset = dataset.map(lambda *windows: tf.stack(windows))  # there're n windows in one take, so use * to pack them
                                                                   # after this step, one take will yield (32, None)
        dataset = dataset.repeat().map(lambda windows: (windows[:, :-1], windows[:, 1:]))
        dataset = dataset.map(
            lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
        dataset = dataset.prefetch(1)
        return dataset

    if batch_size == 1:
        dataset = method_1()
    else:
        dataset = method_2()

    return (tokenizer, max_id, dataset_size, train_size, batch_size), dataset.prefetch(1)


def create_model(max_id, batch_size):
    model = keras.models.Sequential([
        keras.layers.GRU(128, return_sequences=True, stateful=True,
                         dropout=0.2, recurrent_dropout=0.2,
                         batch_input_shape=[batch_size, None, max_id]),
        keras.layers.GRU(128, return_sequences=True, stateful=True,
                         dropout=0.2, recurrent_dropout=0.2),
        keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                        activation="softmax"))
    ])  # need batch_input_shape because the model needs to know how many 'states' it should save

    return model


class ResetStatesCallback(keras.callbacks.Callback):
    """ Before each epoch begin, reset the states, i.e. the first input should not see
        any hidden state from last input in the previous epoch"""
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()


def create_stateless_model(stateful_model, max_id):
    """ Need a new model for different batch size for predicting"""
    stateless_model = keras.models.Sequential([
        keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id]),
        keras.layers.GRU(128, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                        activation="softmax"))
    ])
    stateless_model.build(input_shape=tf.TensorShape([None, max_id]))
    stateless_model.set_weights(stateful_model.get_weights())
    return stateless_model


def preprocess_input(text, tokenizer, max_id):
    """ Preprocess a input for predicting, the input should be a string ["abcdefg"] """
    X = np.array(tokenizer.texts_to_sequences(text)) - 1
    encoded = tf.one_hot(X, depth=max_id)
    return encoded


def predict_next_char(text, tokenizer, max_id, model):
    """ Given a string, predict the next character """
    X_new = preprocess_input([text], tokenizer, max_id)
    Y_pred = model.predict_classes(X_new)
    pred_char = tokenizer.sequences_to_texts(Y_pred + 1)[0][-1]
    return pred_char


def predict_next_oneofmany(text, tokenizer, max_id, model, temperature=1):
    """ Given a string, predict the next character """
    X_new = preprocess_input([text], tokenizer, max_id)
    y_proba = model.predict(X_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]


def complete_text(text, next_len, tokenizer, max_id, model):
    next_text = text
    for i in range(next_len):
        next_text += predict_next_oneofmany(next_text, tokenizer, max_id, model)
    return next_text


def main():
    shakespeare_text = load_shakespeare()   # This is a string

    data_info, dataset = prepare_dataset(shakespeare_text, batch_size=32, n_steps=100)
    tokenizer, max_id, dataset_size, train_size, batch_size = data_info

    model = create_model(max_id, batch_size)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    # history = model.fit(dataset, epoch=2, steps_per_epoch=train_size // batch_size, callback=[ResetStatesCallback])

    stateless_model = create_stateless_model(model, max_id)
    print(complete_text("t", 5, tokenizer, max_id, stateless_model))




if __name__ == "__main__":
    main()
