import tensorflow as tf
from tensorflow import keras
from load_data import load_california_housing_unscaled
from sklearn.preprocessing import StandardScaler
import numpy as np


def get_look_up_table(vocab, num_oov_buckets):
    """ create a look up table from vocabulary """
    indices = tf.range(len(vocab), dtype=tf.int64)
    table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
    num_oov_buckets = 1
    table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)

    return table


def manual_embedding(vocab, num_oov_buckets, embedding_dim):
    table = get_look_up_table(vocab, num_oov_buckets=num_oov_buckets)

    embed_init = tf.random.uniform([len(vocab) + num_oov_buckets, embedding_dim])
    embedding_matrix = tf.Variable(embed_init)

    categories = tf.constant(["NEAR BAY", "DESERT", "INLAND", "INLAND"])
    cat_indices = table.lookup(categories)
    cat_embedding = tf.nn.embedding_lookup(embedding_matrix, cat_indices)

    print(cat_embedding)


def create_model(table):
    """ A model with embedding layer that can process text features """
    regular_inputs = keras.layers.Input(shape=[8])
    categories = keras.layers.Input(shape=[], dtype=tf.string)
    cat_indices = keras.layers.Lambda(lambda cats: table.lookup(cats))(categories)
    cat_embed = keras.layers.Embedding(input_dim=6, output_dim=2)(cat_indices)
    encoded_inputs = keras.layers.concatenate([regular_inputs, cat_embed])
    outputs = keras.layers.Dense(1)(encoded_inputs)
    model = keras.models.Model(inputs=[regular_inputs, categories],
                               outputs=[outputs])

    return model


def main():
    vocab = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
    manual_embedding(vocab, num_oov_buckets=1, embedding_dim=2)

    # TODO: create model and train the model








if __name__ == "__main__":
    main()