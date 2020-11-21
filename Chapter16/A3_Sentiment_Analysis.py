import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from collections import Counter


def preprocess(x_batch, y_batch):
    x_batch = tf.strings.substr(x_batch, pos=0, len=300)  # crop string, only retain the first 300 characters
    x_batch = tf.strings.regex_replace(x_batch, b"<br\\s*/?>", b" ")  # replace any characters other than letters and quotes with spaces
    x_batch = tf.strings.regex_replace(x_batch, b"[^a-zA-Z']", b" ")  # splits the reviews by the spaces
    x_batch = tf.strings.split(x_batch)
    return x_batch.to_tensor(default_value=b"<pad>"), y_batch


def prepare_training_set(vocab_size=10000, num_oov_buckets=1000, batch_size=32):
    """ Generate a training set ready to be fed in a RNN model """
    def encode_words(X_batch, y_batch):
        return table.lookup(X_batch), y_batch

    datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)

    datasets = datasets["train"].batch(batch_size).map(preprocess)
    vocabulary = Counter()
    for x_batch, y_batch in datasets:  # x_batch Tensor[32, 60], i.e. [batch_size, num_words]
        for review in x_batch:  # x_batch Tensor[60,], i.e. [batch_size, num_words]
            vocabulary.update(review.numpy())  # feed an iterable

    truncated_vocabulary = [
        word for word, count in vocabulary.most_common()[:vocab_size]]  # take only the 10000 most common words in vocab

    words = tf.constant(truncated_vocabulary)  # all 10000 words
    word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)  # index 0 - 9999
    vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
    table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)  # look up table created

    datasets = datasets.map(encode_words).prefetch(1)

    return datasets


def create_model(num_id, embed_size=128):
    model = keras.models.Sequential([
        keras.layers.Embedding(num_id, embed_size,
                               input_shape=[None]),
        keras.layers.GRU(128, return_sequences=True),
        keras.layers.GRU(128),
        keras.layers.Dense(1, activation="sigmoid")
    ])  # sigmoid corresponds to 0-1 label
    return model


def create_model_using_masking(num_id, embed_size=128):
    """ Mask out the <pad> """
    K = keras.backend
    inputs = keras.layers.Input(shape=[None])
    mask = keras.layers.Lambda(lambda inputs: K.not_equal(inputs, 0))(inputs)  # <pad> is encoded to 0
    z = keras.layers.Embedding(num_id, embed_size)(inputs)
    z = keras.layers.GRU(128, return_sequences=True)(z, mask=mask)
    z = keras.layers.GRU(128)(z, mask=mask)
    outputs = keras.layers.Dense(1, activation="sigmoid")(z)
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model


def main():
    vocab_size = 10000
    num_oov_buckets = 1000
    batch_size = 32
    train_set = prepare_training_set(vocab_size, num_oov_buckets, batch_size)

    num_id = vocab_size + num_oov_buckets
    model = create_model_using_masking(num_id)

    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    history = model.fit(train_set, epochs=1)


if __name__ == "__main__":
    main()
