import numpy as np


def mae_image(dt_image: np.array, approximate_dt_image: np.array) -> float:
    """
    The average of the difference between the two images is computed

    :param dt_image:
    :param approximate_dt_image:
    :return: The mean absolute error
    """

    return np.mean(abs(dt_image - approximate_dt_image))

