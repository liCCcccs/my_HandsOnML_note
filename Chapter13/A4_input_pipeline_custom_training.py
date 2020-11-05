import tensorflow as tf
from tensorflow import keras
from load_data import load_california_housing_unscaled
from sklearn.preprocessing import StandardScaler

Train_filepaths = ['datasets/housing/my_train_00.csv', 'datasets/housing/my_train_01.csv',
                   'datasets/housing/my_train_02.csv', 'datasets/housing/my_train_03.csv',
                   'datasets/housing/my_train_04.csv', 'datasets/housing/my_train_05.csv',
                   'datasets/housing/my_train_06.csv', 'datasets/housing/my_train_07.csv',
                   'datasets/housing/my_train_08.csv', 'datasets/housing/my_train_09.csv',
                   'datasets/housing/my_train_10.csv', 'datasets/housing/my_train_11.csv',
                   'datasets/housing/my_train_12.csv', 'datasets/housing/my_train_13.csv',
                   'datasets/housing/my_train_14.csv', 'datasets/housing/my_train_15.csv',
                   'datasets/housing/my_train_16.csv', 'datasets/housing/my_train_17.csv',
                   'datasets/housing/my_train_18.csv', 'datasets/housing/my_train_19.csv']

Valid_filepaths = ['datasets/housing/my_valid_00.csv', 'datasets/housing/my_valid_01.csv',
                    'datasets/housing/my_valid_02.csv', 'datasets/housing/my_valid_03.csv',
                    'datasets/housing/my_valid_04.csv', 'datasets/housing/my_valid_05.csv',
                    'datasets/housing/my_valid_06.csv', 'datasets/housing/my_valid_07.csv',
                    'datasets/housing/my_valid_08.csv', 'datasets/housing/my_valid_09.csv']

Test_filepaths = ['datasets/housing/my_test_00.csv', 'datasets/housing/my_test_01.csv',
                  'datasets/housing/my_test_02.csv', 'datasets/housing/my_test_03.csv',
                  'datasets/housing/my_test_04.csv', 'datasets/housing/my_test_05.csv',
                  'datasets/housing/my_test_06.csv', 'datasets/housing/my_test_07.csv',
                  'datasets/housing/my_test_08.csv', 'datasets/housing/my_test_09.csv']


def get_preprocess(n_inputs, X_mean, X_std):
    def preprocess(line):
        defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
        fields = tf.io.decode_csv(line, record_defaults=defs)
        x = tf.stack(fields[:-1])
        y = tf.stack(fields[-1:])
        return (x - X_mean) / X_std, y
    return preprocess


def get_mean_std(data):
    scaler = StandardScaler()
    scaler.fit(data)
    X_mean = scaler.mean_
    X_std = scaler.scale_

    return X_mean, X_std


def csv_reader_dataset(filepaths, preprocess_fn, repeat=1, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess_fn, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)


def create_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=input_shape),
        keras.layers.Dense(1),
    ])

    return model


def custom_training(model, train_set, X_train_len):
    optimizer = keras.optimizers.Nadam(learning_rate=0.01)
    loss_fn = keras.losses.mean_squared_error

    n_epochs = 5
    batch_size = 32
    n_steps_per_epoch = X_train_len // batch_size
    total_steps = n_epochs * n_steps_per_epoch
    global_step = 0
    for X_batch, y_batch in train_set.take(total_steps):
        # one take is a tensor of shape (batch_size, num_input)
        global_step += 1
        print("\rGlobal step {}/{}".format(global_step, total_steps), end="")
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_california_housing_unscaled()
    X_mean, X_std = get_mean_std(X_train)

    n_inputs = 8
    preprocess_fn = get_preprocess(n_inputs, X_mean, X_std)

    train_set = csv_reader_dataset(Train_filepaths, preprocess_fn, repeat=None)  # Put Training set into pipeline
    valid_set = csv_reader_dataset(Valid_filepaths, preprocess_fn)
    test_set = csv_reader_dataset(Test_filepaths, preprocess_fn)

    model = create_model(X_train.shape[1:])

    custom_training(model, train_set, len(X_train))


if __name__ == "__main__":
    main()
