import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_sample_image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_image(image, isColored=True):
    """ pixel value should be scaled to 0-1 """
    if isColored:
        plt.imshow(image, interpolation="nearest")
    else:
        plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.show()


def main():
    model = keras.applications.resnet50.ResNet50(weights="imagenet")

    cat = plt.imread("./sample_img/originalLeopard.jpg")  # a local image
    crop_box = [0.3, 0.2, 1, 0.8]  # [up, left, down, right]
    images = np.array([cat])  # need to be batched
    images_resized = tf.image.crop_and_resize(images, [crop_box], [0], [224, 224])
    #images_resized = tf.image.crop_and_resize(images, [china_box, flower_box], [0, 1], [224, 224])  # for multiple images

    plot_image(images_resized[0] / 255, isColored=True)

    inputs = keras.applications.resnet50.preprocess_input(images_resized)  # input need 0-255 ranged image
    Y_proba = model.predict(inputs)

    top_K = keras.applications.resnet50.decode_predictions(Y_proba, top=3)
    for image_index in range(len(images)):
        print("Image #{}".format(image_index))
        for class_id, name, y_proba in top_K[image_index]:
            print("  {} - {:12s} {:.2f}%".format(class_id, name, y_proba * 100))
        print()


if __name__ == "__main__":
    main()
