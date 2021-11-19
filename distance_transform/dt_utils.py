import random

import numpy as np
import matplotlib.pyplot as plt
from distance_transform.wave_propagation import wave_propagation_dt_gmap
from combinatorial.pixelmap import LabelMap
from distance_transform.wave_propagation import generate_accumulation_directions_cell
from combinatorial.utils import *


def plot_binary_image(image: np.array) -> None:
    """

    :param image: All the values have to be ones or zeros
    :return:
    """

    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.show()


def plot_dt_image(image: np.array, max_value = None) -> None:
    if max_value is None:
        # find max value
        max_value = -1
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] > max_value:
                    max_value = image[i][j]

    if max_value > 255:
        raise Exception(f"max value: {max_value} is greater than 255")

    plt.imshow(image, cmap="gray", vmin=0, vmax=max_value)
    plt.show()


def generate_random_binary_image(image_size: int, background_pixel_probability: float) -> np.array:
    image = np.ones((image_size, image_size), dtype=int)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            random_float = random.random()
            if random_float < background_pixel_probability:
                image[i][j] = 0

    return image


def gmap_dt_equal(gmap_1, gmap_2) -> bool:
    """
    It returns True if each dart in the two gmap passed as parameters has the same distance value.

    PRECONDITIONS:
    The two gmaps need to have the same structure.
    """
    for dart_1, dart_2 in zip(gmap_1.darts, gmap_2.darts):
        if dart_1 != dart_2:
            raise Exception(f"The identifiers don't match. id 1: {dart_1} - id 2: {dart_2}")

        distance_1 = gmap_1.distances[dart_1]
        distance_2 = gmap_2.distances[dart_2]
        if distance_1 != distance_2:
            print(f"dart id: {dart_1} - dart 1 distance: {distance_1} - dart 2 distance: {distance_2}")
            return False

    return True


def compute_dt_reduction(image, reduction_factor: float, show_gmap: bool = True,
                        build_image_interpolate: bool = True):

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

        # plot
        if show_gmap:
            gmap.plot_dt(fill_cell='face')

        dt_image = build_dt_grey_image(gmap, interpolate_missing_values=build_image_interpolate)
        print("image successfully retrieved")
        plot_dt_image(dt_image, None)