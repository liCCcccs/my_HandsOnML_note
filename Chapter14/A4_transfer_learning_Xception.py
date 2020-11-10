import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from functools import partial
import matplotlib.pyplot as plt


def central_crop(image):
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]])
    top_crop = (shape[0] - min_dim) // 4
    bottom_crop = shape[0] - top_crop  # make top and bottom symmetric
    left_crop = (shape[1] - min_dim) // 4
    right_crop = shape[1] - left_crop  # make left and right symmetric
    return image[top_crop:bottom_crop, left_crop:right_crop]


def random_crop(image):
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]]) * 90 // 100  # output is still int, cropped image is 90% large
    return tf.image.random_crop(image, [min_dim, min_dim, 3])


def preprocess(image, label, randomize=False):
    if randomize:
        cropped_image = random_crop(image)
        cropped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        cropped_image = central_crop(image)
    resized_image = tf.image.resize(cropped_image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label


def main():
    test_set_raw, valid_set_raw, train_set_raw = tfds.load(
        "tf_flowers",
        split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
        as_supervised=True)

    batch_size = 32
    train_set = train_set_raw.shuffle(1000).repeat()
    train_set = train_set.map(partial(preprocess, randomize=True)).batch(batch_size).prefetch(1)
    valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
    test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)

    class_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']
    n_classes = len(class_names)
    dataset_size = 3670

    base_model = keras.applications.xception.Xception(weights="imagenet",
                                                      include_top=False)
    avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(n_classes, activation="softmax")(avg)
    model = keras.models.Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False

    base_model.summary()

    optimizer = keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, decay=0.01)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["sparse_categorical_accuracy"])

    history = model.fit(train_set,
                        steps_per_epoch=int(0.75 * dataset_size / batch_size),
                        validation_data=valid_set,
                        validation_steps=int(0.15 * dataset_size / batch_size),
                        epochs=5)  # since infinitely repeat on dataset, need to set steps_per_epoch here


if __name__ == "__main__":
    main()
