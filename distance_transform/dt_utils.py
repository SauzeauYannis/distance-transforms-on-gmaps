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
