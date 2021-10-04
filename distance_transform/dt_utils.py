import random

import numpy as np
import matplotlib.pyplot as plt


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
