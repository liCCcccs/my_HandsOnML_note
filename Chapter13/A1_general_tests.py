import tensorflow as tf
from tensorflow import keras


def main():
    X = tf.range(10)
    dataset = tf.data.Dataset.from_tensor_slices(X)  # A set of 10 tensors 0-9
    dataset = dataset.repeat(3)  # A set of 30 tensors, 0-9 0-9 0-9
    dataset = dataset.batch(6)  # A set of 5 tensors, each tensor is a batch of length 6
    dataset = dataset.map(lambda x: x * 2, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # every element x2
    dataset = dataset.unbatch()
    dataset = dataset.filter(lambda x: x < 10)  # can only apply to unbatched data
    dataset = dataset.shuffle(buffer_size=3, seed=42).batch(7)
    for item in dataset:
        print(item)


if __name__ == "__main__":
    main()

