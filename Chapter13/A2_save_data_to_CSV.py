import tensorflow as tf
from tensorflow import keras
from load_data import load_california_housing_unscaled
import os
import numpy as np


def save_to_multiple_csv_files(data, name_prefix, header=None, n_parts=10):
    """ Separate `data` into `n_parts`, save them in `n_parts` CSV files """
    housing_dir = os.path.join("datasets", "housing")  # on Linux: "./datasets/housing"
    os.makedirs(housing_dir, exist_ok=True)
    path_format = os.path.join(housing_dir, "my_{}_{:02d}.csv")

    filepaths = []
    m = len(data)
    for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filepaths.append(part_csv)
        with open(part_csv, "wt", encoding="utf-8") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([repr(col) for col in data[row_idx]]))  # repr(col): make a printable version of col
                f.write("\n")
    return filepaths


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test, feature_names = load_california_housing_unscaled(get_feature_name=True)

    train_data = np.c_[X_train, y_train]  # np.c_(): concatenation along the second axis
    valid_data = np.c_[X_valid, y_valid]
    test_data = np.c_[X_test, y_test]
    header_cols = feature_names + ["MedianHouseValue"]
    header = ",".join(header_cols)

    train_filepaths = save_to_multiple_csv_files(train_data, "train", header, n_parts=20)
    valid_filepaths = save_to_multiple_csv_files(valid_data, "valid", header, n_parts=10)
    test_filepaths = save_to_multiple_csv_files(test_data, "test", header, n_parts=10)

    print(train_filepaths)
    print(valid_filepaths)
    print(test_filepaths)


if __name__ == "__main__":
    main()
