import tensorflow as tf
from tensorflow import keras
from load_data import load_california_housing_unscaled
from sklearn.preprocessing import StandardScaler
import numpy as np


def download_housing_with_text():
    import os
    import tarfile
    import urllib

    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    HOUSING_PATH = os.path.join("datasets", "housing_with_text")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    housing_url = HOUSING_URL
    housing_path = HOUSING_PATH

    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def csv_reader_dataset(filepaths, repeat=1, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)


def manual_encode_category():
    vocab = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
    indices = tf.range(len(vocab), dtype=tf.int64)
    table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
    num_oov_buckets = 2
    table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)

    categories = tf.constant(
        ["NEAR BAY", "DESERT", "INLAND", "INLAND", "DESERT", "ALL", "DESERT"])
    cat_indices = table.lookup(categories)
    cat_one_hot = tf.one_hot(cat_indices, depth=len(vocab) + num_oov_buckets)

    print(cat_one_hot)


def main():
    #download_housing_with_text()

    train_set = csv_reader_dataset(["./datasets/housing_with_text/housing.csv"])

    # TODO: play with this dataset, train a model using one-hot categorical input








if __name__ == "__main__":
    main()
