import typing

import numpy as np


def mae_image(dt_image: np.array, approximate_dt_image: np.array) -> typing.Any:
    """The average of the difference between the two images is computed

    Args:
        dt_image: the distance transform image
        approximate_dt_image: the approximate distance transform image

    Returns:
        float: the average of the difference between the two images
    """

    return np.mean(abs(dt_image - approximate_dt_image))
