import os
import argparse
import pickle
import logging
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid


logger = logging.getLogger(__name__)


def parse_args():
    arg_parser = argparse.ArgumentParser(
        "Perform dataset analysis on .npy matrices of images and labels.")
    arg_parser.add_argument(
        "dataset_npy_dir_path", type=str,
        help="Path to the directory which holds results of prepare_dataset.py.")
    arg_parser.add_argument(
        "metadata_path", type=str,
        help="Path to the file which holds metadata of the dataset along with "
             "mapping of int labels to str labels.")
    arg_parser.add_argument(
        "out_dir_path", type=str,
        help="Path to the directory which will hold output artifacts such as "
             "plot images and analysis log.")

    args = arg_parser.parse_args()
    return args


def calc_mean_saturations(images):
    """ Calculate mean saturations image-wise. """
    saturations = np.max(images, axis=-1) - np.min(images, axis=-1)
    mean_saturations = np.mean(saturations, axis=(-2, -1))
    return mean_saturations


NPY_IMAGES_FILENAME = "images.npy"
NPY_LABELS_FILENAME = "labels.npy"


def load_dataset(dataset_npy_dir_path):
    images = np.load(os.path.join(dataset_npy_dir_path, NPY_IMAGES_FILENAME))
    labels = np.load(os.path.join(dataset_npy_dir_path, NPY_LABELS_FILENAME))
    return images, labels


def show_images_sequential(images, limit):
    """ Show images until mouse key is pressed/limit of images reached. """
    for i in range(images.shape[0]):
        plt.imshow(images[i])

        # End on mousepress. Continue on keypress.
        if not plt.waitforbuttonpress():
            break

        # End when images limit is reaches.
        if i >= limit:
            break


def show_images_grid(images, images_per_row, title=None, figsize=None,
                     save_fig=False, save_fig_path=None):
    """ Show images on one plot in grid form with optional figure save. """
    images_tensor = torch.tensor(np.transpose(images, (0, 3, 1, 2)))
    grid = np.transpose(
        make_grid(images_tensor, nrow=images_per_row), (1, 2, 0))
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(grid)
    plt.waitforbuttonpress()
    if save_fig:
        plt.savefig(save_fig_path)
    plt.close()


MEAN_SATURATION_THRESHOLD = 1


def find_grayscale_images(images):
    """ Find images with mean saturation less than the set threshold. """
    mean_saturations = calc_mean_saturations(images)
    indices = np.where(mean_saturations <= MEAN_SATURATION_THRESHOLD)[0]
    return indices


def find_coloured_images(images):
    """ Find images with mean saturation bigger than the set threshold. """
    mean_saturations = calc_mean_saturations(images)
    indices = np.where(mean_saturations >= MEAN_SATURATION_THRESHOLD)[0]
    return indices


def discard_grayscale_images(images, labels):
    """
    Filter out images that are suspected to be grayscale because of their small
    mean saturation.
    """
    indices = find_coloured_images(images)
    images = images[indices]
    labels = labels[indices]
    return images, labels


OUT_LOG_FILENAME = "log.txt"


def setup_logger(out_dir_path):
    """ Setup file and console handlers for logger """
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(
        os.path.join(out_dir_path, OUT_LOG_FILENAME), "w")
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(console_handler)


# Filenames for output artifacts.
DATASET_DISTRIBUTION_PLOT_FILENAME = "dataset_distribution.png"
DATASET_SAMPLE_PLOT_FILENAME = "dataset_sample.png"
DATASET_CLASSES_SAMPLE_PLOT_FILENAME = "dataset_classes_sample.png"
MEAN_IMAGE_PLOT_FILENAME = "mean_image.png"
MEAN_CLASS_IMAGE_PLOT_FILENAME = "mean_{}_image.png"
MEAN_CLASSES_IMAGES_PLOT_FILENAME = "mean_classes_images.png"
SATURATION_HISTOGRAM_PLOT_FILENAME = "saturation_histogram.png"
GRAYSCALE_CANDIDATES_SAMPLE_PLOT_FILENAME = "grayscale_candidates.png"
FILTERED_DATASET_DISTRIBUTION_PLOT_FILENAME = "filtered_dataset_distribution" \
                                              ".png"
FILTERED_SATURATION_HISTOGRAM_PLOT_FILENAME = "filtered_saturation_histogram" \
                                              ".png"
FILTERED_SAMPLE_PLOT_FILENAME = "filtered_dataset_sample.png"


def main():
    args = parse_args()

    os.makedirs(args.out_dir_path, exist_ok=True)

    setup_logger(args.out_dir_path)

    with open(args.metadata_path, "rb") as file:
        metadata = pickle.load(file, encoding="bytes")
        labels_mapping = [bytes_label_name.decode("ascii")
                          for bytes_label_name in metadata[b"label_names"]]

    images, labels = load_dataset(args.dataset_npy_dir_path)

    # Randomize images with manually specified random seed for reproducibility.
    np.random.seed(1000)
    shuffling_indices = np.random.permutation(images.shape[0])
    images = images[shuffling_indices]
    labels = labels[shuffling_indices]

    # Get basic dataset parameters
    samples_count = images.shape[0]
    unique_labels = np.unique(labels)
    mapped_unique_labels = [labels_mapping[label] for label in unique_labels]
    classes_count = unique_labels.shape[0]
    classes_counts = {}
    for label in unique_labels:
        classes_counts[label] = np.where(labels == label)[0].shape[0]

    logger.info("Basic dataset analysis:")
    logger.info("  - Samples count: {}".format(samples_count))
    logger.info("  - Classes count: {}".format(classes_count))
    logger.info("  - Set of unique labels: {}".format(mapped_unique_labels))
    logger.info("  - Counts of samples per class:")
    for label, count in classes_counts.items():
        logger.info("    - {}: {}".format(labels_mapping[label], count))
    logger.info("")

    # Show distribution of samples in the dataset.
    plt.title("Distribution of samples per class")
    classes_counts_keys = list(classes_counts.keys())
    plt.xticks(rotation=45)
    plt.bar(
        [labels_mapping[key] for key in classes_counts_keys],
        [classes_counts[key] for key in classes_counts_keys])
    plt.tight_layout()
    plt.waitforbuttonpress()
    plt.savefig(
        os.path.join(args.out_dir_path, DATASET_DISTRIBUTION_PLOT_FILENAME))
    plt.close()

    # Show sample of 100 images from the dataset.
    show_images_grid(
        images[:100], 10, title="Sample of images from CIFAR-10",
        figsize=(8, 8), save_fig=True,
        save_fig_path=os.path.join(
            args.out_dir_path, DATASET_SAMPLE_PLOT_FILENAME))

    # Show sample of 100 images sorted by their class in rows.
    classes_indices = []
    for label in unique_labels:
        classes_indices.append(np.where(labels == label)[0][:10])
    classes_indices = np.concatenate(classes_indices)
    show_images_grid(
        images[classes_indices], 10,
        title="Sample of images from CIFAR-10 sorted by class",
        figsize=(8, 8), save_fig=True,
        save_fig_path=os.path.join(
            args.out_dir_path, DATASET_CLASSES_SAMPLE_PLOT_FILENAME))

    # Save mean image - GAN might collapse to it.
    mean_image = np.mean(images, axis=0).astype(np.uint8)
    plt.figure(figsize=(4, 4))
    plt.title("Mean of all images in the dataset")
    plt.imshow(mean_image)
    plt.waitforbuttonpress()
    plt.savefig(
        os.path.join(args.out_dir_path, MEAN_IMAGE_PLOT_FILENAME))
    plt.close()

    # Save other mean images - GAN might collapse to them too.
    for label in unique_labels:
        mean_class_image = np.mean(
            images[np.where(labels == label)[0]], axis=0).astype(np.uint8)
        plt.figure(figsize=(4, 4))
        plt.title(
            "Mean of all images in the class '{}'"
            .format(labels_mapping[label]))
        plt.imshow(mean_class_image)
        plt.waitforbuttonpress()
        plt.savefig(
            os.path.join(
                args.out_dir_path,
                MEAN_CLASS_IMAGE_PLOT_FILENAME.format(labels_mapping[label])))
        plt.close()

    # Analyze saturations of the images.
    mean_saturations = calc_mean_saturations(images)
    plt.title("Histogram of saturations of the images")
    plt.hist(mean_saturations, bins=200)
    plt.waitforbuttonpress()
    plt.savefig(
        os.path.join(args.out_dir_path, SATURATION_HISTOGRAM_PLOT_FILENAME))
    plt.close()

    potential_grayscale_indices = find_grayscale_images(images)
    potential_grayscales_count = potential_grayscale_indices.shape[0]

    logger.info(
        "Number of potential grayscale images: {} found with threshold: {}."
        .format(potential_grayscales_count, MEAN_SATURATION_THRESHOLD))

    # Show up to 100 grayscale candidates.
    show_images_grid(
        images[potential_grayscale_indices][:100], 10,
        title="Sample of potentially grayscale images from CIFAR-10",
        figsize=(8, 8), save_fig=True,
        save_fig_path=os.path.join(
            args.out_dir_path, GRAYSCALE_CANDIDATES_SAMPLE_PLOT_FILENAME))

    # Analyze filtered dataset.
    filtered_images, filtered_labels = discard_grayscale_images(images, labels)
    filtered_images_count = filtered_images.shape[0]
    filtered_classes_counts = {}
    for label in unique_labels:
        filtered_classes_counts[label] = np.where(
            filtered_labels == label)[0].shape[0]

    logger.info(
        "Analysis of the dataset filtered with threshold {}:"
        .format(MEAN_SATURATION_THRESHOLD))
    logger.info("  - Samples count: {}".format(filtered_images_count))
    logger.info("  - Counts of samples per class:")
    for label, count in filtered_classes_counts.items():
        logger.info("    - {}: {}".format(labels_mapping[label], count))

    # Show distribution of samples in the filtered dataset.
    plt.title("Distribution of filtered samples per class")
    filtered_classes_counts_keys = list(filtered_classes_counts.keys())
    plt.xticks(rotation=45)
    plt.bar(
        [labels_mapping[key] for key in filtered_classes_counts_keys],
        [filtered_classes_counts[key] for key in filtered_classes_counts_keys])
    plt.tight_layout()
    plt.waitforbuttonpress()
    plt.savefig(
        os.path.join(
            args.out_dir_path, FILTERED_DATASET_DISTRIBUTION_PLOT_FILENAME))
    plt.close()

    # Analyze saturations of the filtered images.
    mean_filtered_saturations = calc_mean_saturations(filtered_images)
    plt.title("Histogram of saturations of the filtered images")
    plt.hist(mean_filtered_saturations, bins=200)
    plt.waitforbuttonpress()
    plt.savefig(
        os.path.join(
            args.out_dir_path, FILTERED_SATURATION_HISTOGRAM_PLOT_FILENAME))
    plt.close()

    # Plot 400 samples for final anomalies detection.
    show_images_grid(
        filtered_images[:400], 20,
        title="Sample of filtered images from CIFAR-10",
        figsize=(8, 8), save_fig=True,
        save_fig_path=os.path.join(
            args.out_dir_path, FILTERED_SAMPLE_PLOT_FILENAME))


if __name__ == "__main__":
    main()
