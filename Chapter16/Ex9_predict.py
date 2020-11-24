import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
from functools import partial

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

INPUT_CHARS = "".join(sorted(set("".join(MONTHS)))) + "01234567890, "
TARGET_CHARS = "".join(sorted(set("1234567890-")))


def id_to_char(ids, chars=INPUT_CHARS):
    return "".join([chars[i - 1] for i in ids if i != 0])


def char_to_id(s, chars=INPUT_CHARS):
    return [chars.index(char) for char in s]


def preprocess_input(s):
    s_id = char_to_id(s, chars=INPUT_CHARS)
    s_id = np.array(s_id) + 1
    while len(s_id) != 18:
        s_id = np.append(s_id, 0)  # pad with zero to keep all input sequence the same length as the longest sequence
    # the model needs input shape [batch_size, sequence_length], in this case we make batch_size = 1
    return np.expand_dims(s_id, axis=0)


def main():
    model = keras.models.load_model("./saved_model/Ex9_simple_encoder_decoder.h5")

    X_test = preprocess_input("September 17, 2009")
    print(X_test)
    y_test = model.predict_classes(X_test)
    y_decoded = id_to_char(y_test[0], chars=TARGET_CHARS)
    print(y_decoded)

    X_test = preprocess_input("September 30, 2020")
    y_test = model.predict_classes(X_test)
    y_decoded = id_to_char(y_test[0], chars=TARGET_CHARS)
    print(y_decoded)

    X_test = preprocess_input("February 2, 1998")
    y_test = model.predict_classes(X_test)
    y_decoded = id_to_char(y_test[0], chars=TARGET_CHARS)
    print(y_decoded)


if __name__ == "__main__":
    main()
