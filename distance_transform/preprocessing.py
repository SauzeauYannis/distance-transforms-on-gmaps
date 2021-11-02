import typing

import numpy as np


def find_borders(labeled_image: np.array, label_value: int) -> np.array:
    """
    A pixel of a labeled component of interest is considered part of the border
    if the label at least one of his neighbours is different from his label.
    4-neighbourhood is considered.

    I can consider a different rule to find the borders.

    0: border
    255: foreground
    100: inside
    """

    def is_equal_to_neighbours(image: np.array, value_position: typing.Tuple[int, int]) -> True:
        value_x = value_position[0]
        value_y = value_position[1]
        x_values = [value_x, value_x - 1, value_x, value_x + 1]
        y_values = [value_y -1, value_y, value_y + 1, value_y]
        for x, y in zip(x_values, y_values):
            if 0 < x < image.shape[0] and 0 < y < image.shape[1] and image[x][y] != image[value_x][value_y]:
                return False
        return True

    image = np.copy(labeled_image)

    for i in range(labeled_image.shape[0]):
        for j in range(labeled_image.shape[1]):
            curr_pixel = labeled_image[i][j]
            if curr_pixel != label_value:
                image[i][j] = 255 # foreground
            elif is_equal_to_neighbours(labeled_image, (i, j)):
                image[i][j] = 200 # inside
            else:
                image[i][j] = 0 # border

    return image
