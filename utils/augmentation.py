import numpy as np
import os
import torch
from torchvision.utils import make_grid
from matplotlib import pyplot as plt


NPY_IMAGES_FILENAME = "images.npy"
NPY_LABELS_FILENAME = "labels.npy"


def load_dataset(dataset_npy_dir_path):
    images = np.load(os.path.join(dataset_npy_dir_path, NPY_IMAGES_FILENAME))
    labels = np.load(os.path.join(dataset_npy_dir_path, NPY_LABELS_FILENAME))
    return images, labels


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


def flip(image):
    return image[:, ::-1]


def translate(image, max_offset):
    padded_image = np.pad(
        image, ((max_offset, max_offset), (max_offset, max_offset), (0, 0)),
        "edge")
    offset_x = np.random.randint(max_offset * 2 + 1)
    offset_y = np.random.randint(max_offset * 2 + 1)
    return padded_image[offset_x:offset_x + 32, offset_y:offset_y + 32]


def main():
    images, labels = load_dataset("../data/dataset_npy")
    plt.figure(figsize=(4, 4))
    plt.title("Example image from CIFAR-10 dataset")
    plt.imshow(images[19])
    plt.waitforbuttonpress()
    augmented_images = []
    for i in range(16):
        to_flip = np.random.randint(2)
        img = images[19]
        if to_flip == 0:
            img = flip(images[19])
        img = translate(img, 3)
        augmented_images.append(img)
    show_images_grid(
        augmented_images, 4, figsize=(8, 8),
        title="Example augmentation of random image from CIFAR-10")


if __name__ == "__main__":
    main()
