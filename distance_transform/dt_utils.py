import random

import numpy as np
import matplotlib.pyplot as plt

from combinatorial.pixelmap import LabelMap
from combinatorial.utils import build_dt_grey_image_from_gmap
from distance_transform.wave_propagation import wave_propagation_dt_gmap, generate_accumulation_directions_cell


def plot_binary_image(image: np.array) -> None:
    """Plot a binary image

    Args:
        image: the binary image to plot
    """

    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.show()


def plot_dt_image(image: np.array, max_value: int = None) -> None:
    """Plot a distance transform image

    Args:
        image: the distance transform image to plot
        max_value: the maximum value of the distance transform image
    """

    if max_value is None:
        max_value = image.max()

    if max_value > 255:
        raise Exception(f"max value: {max_value} is greater than 255")

    plt.imshow(image, cmap="gray", vmin=0, vmax=max_value)
    plt.show()


def plot_color_image(image: np.array) -> None:
    """Plot a color image

    Args:
        image: the color image to plot
    """

    plt.imshow(image)
    plt.show()


def generate_random_binary_image(image_size: int, background_pixel_probability: float) -> np.array:
    """Generate a binary image with random pixels

    Args:
        image_size: the size of the image
        background_pixel_probability: the probability of a pixel to be background between 0 and 1

    Returns:
        np.array: the binary image generated
    """

    image = np.zeros((image_size, image_size))
    for i in range(image_size):
        for j in range(image_size):
            image[i][j] = random.random() < background_pixel_probability

    return image


def gmap_dt_equal(gmap_1, gmap_2) -> bool:
    """Compare two gmaps distance transform

    Args:
        gmap_1: the first gmap
        gmap_2: the second gmap

    Returns:
        bool: True if the distance transform of the two gmaps are equal

    Precondition:
        gmap_1 and gmap_2 must have the same structure
    """
    for dart_1, dart_2 in zip(gmap_1.darts, gmap_2.darts):
        if dart_1 != dart_2:
            print(
                f"The identifiers don't match. id 1: {dart_1} - id 2: {dart_2}")
            return False

        distance_1 = gmap_1.distances[dart_1]
        distance_2 = gmap_2.distances[dart_2]

        if distance_1 != distance_2:
            print(
                f"dart id: {dart_1} - dart 1 distance: {distance_1} - dart 2 distance: {distance_2}")
            return False

    return True


def compute_dt_reduction(
    image,
    reduction_factor: float,
    show_gmap: bool = True,
    build_image_interpolate: bool = True
) -> None:
    """Compute the distance transform of an image with a reduction factor on

    Args:
        image: the image to compute the distance transform
        reduction_factor: the reduction factor to apply
        show_gmap: if True, the gmap is plotted
        build_image_interpolate: if True, the image is interpolated to build the distance transform
    """

    gmap = LabelMap.from_labels(image)
    print("gmap successfully created")
    gmap.remove_edges(reduction_factor)
    print("edges successfully removed")
    gmap.remove_vertices()
    print("vertices successfully removed")

    if show_gmap:
        gmap.plot()

    accumulation_directions = generate_accumulation_directions_cell(2)
    wave_propagation_dt_gmap(gmap, None, accumulation_directions)
    print("dt successfully computed")

    if show_gmap:
        gmap.plot_dt(fill_cell='face')

    dt_image = build_dt_grey_image_from_gmap(
        gmap, interpolate_missing_values=build_image_interpolate)
    print("image successfully retrieved")
    plot_dt_image(dt_image, None)
