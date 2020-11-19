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
    dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

    window_length = n_steps + 1  # target = input shifted 1 character ahead
    dataset = dataset.repeat().window(window_length, shift=1, drop_remainder=True)

    dataset = dataset.flat_map(lambda window: window.batch(window_length))
    dataset = dataset.shuffle(10000).batch(batch_size)
    a = dataset.take(1)
    dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))  # a window: [batch_size, window_length]
    dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))

    return (tokenizer, max_id, dataset_size, train_size, batch_size), dataset.prefetch(1)


def create_model(max_id):
    model = keras.models.Sequential([
        keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id],
                         dropout=0.2, recurrent_dropout=0.2),
        keras.layers.GRU(128, return_sequences=True,
                         dropout=0.2, recurrent_dropout=0.2),
        keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                        activation="softmax"))
    ])  # one hot coding, so input shape [None, max_id], output the probability of each character at each time step
    return model


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

    model = create_model(max_id)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    # history = model.fit(dataset, steps_per_epoch=train_size // batch_size,
    #                     epochs=10)

    pred_char = predict_next_char("How are yo", tokenizer, max_id,  model)
    print(pred_char)

    next_sentense = complete_text("How", 5, tokenizer, max_id, model)
    print(next_sentense)



if __name__ == "__main__":
    main()
