import pickle
import os
import argparse

import numpy as np


def parse_args():
    arg_parser = argparse.ArgumentParser(
        "Convert CIFAR-10 dataset into 2 .npy files - one with images and one"
        "with labels.")
    arg_parser.add_argument(
        "dataset_raw_dir_path", type=str,
        help="Path to the directory with original CIFAR-10 contents in it.")
    arg_parser.add_argument(
        "dataset_npy_dir_path", type=str,
        help="Path to the directory which will hold output .npy artifacts.")

    args = arg_parser.parse_args()
    return args


BATCHES_FILENAMES = ["data_batch_{}".format(i) for i in range(1, 6)] \
                    + ["test_batch"]


def load_batches(dataset_raw_dir_path):
    """
    Load all batch files from CIFAR-10 dataset dir pointed to in
    dataset_raw_dir_path.
    """
    batches = []
    for filename in BATCHES_FILENAMES:
        batch_file_path = os.path.join(dataset_raw_dir_path, filename)
        with open(batch_file_path, "rb") as file:
            batch = pickle.load(file, encoding="bytes")
        batches.append(batch)

    return batches


def merge_batches(batches_dicts):
    """
    Merge a list of batches dicts into a pair of numpy arrays.

    Args:
        batches_dicts: list of unpickled dictionaries of data batches from
        CIFAR-10 dataset

    Returns:
        A pair of variables: a numpy array of merged image data from all batches
        and a numpy array of merged labels from all batches

    """
    images = []
    labels = []
    for batch in batches_dicts:
        images.append(batch[b"data"])
        labels.append(batch[b"labels"])

    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    # According to CIFAR-10 web page, data inside it is stored in row-major
    # order. First 1024 values is the whole R channel, next 1024 is the whole
    # G channel etc. We change it to standard [batch_size, w, h, n_channels]
    # format.
    images = images.reshape([-1, 3, 32, 32])
    images = np.transpose(images, [0, 2, 3, 1])

    return images, labels


NPY_IMAGES_FILENAME = "images.npy"
NPY_LABELS_FILENAME = "labels.npy"


def save_npy_dataset(dataset_npy_dir_path, images, labels):
    """
    Save images and labels to separate .npy files inside of
    dataset_npy_dir_path.
    """
    os.makedirs(dataset_npy_dir_path, exist_ok=True)

    images_file_path = os.path.join(dataset_npy_dir_path, NPY_IMAGES_FILENAME)
    labels_file_path = os.path.join(dataset_npy_dir_path, NPY_LABELS_FILENAME)

    np.save(images_file_path, images)
    np.save(labels_file_path, labels)


def main():
    args = parse_args()

    batches_dicts = load_batches(args.dataset_raw_dir_path)
    images, labels = merge_batches(batches_dicts)
    save_npy_dataset(args.dataset_npy_dir_path, images, labels)


if __name__ == "__main__":
    main()
