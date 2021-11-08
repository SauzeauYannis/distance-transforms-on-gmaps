import typing

import numpy as np


# Modify the function (maybe change also the name) in order to pass 2 array of labels
# seeds_labels, propagation_labels
def find_borders(labeled_image: np.array, label_value: int) -> np.array:
    """
    A pixel of a labeled component of interest is considered part of the border
    if the label of at least one of his neighbours is different from his label.
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

def find_borders_on_gmap(gmap, label_value: int, face_value):
    """

    It modifies the gmap passed as parameter.

    0: border
    255: foreground
    100: inside

    :param gmap:
    :param label_value:
    :return:
    """

    pass


def clean_borders(image: np.array, kernel_size: int) -> np.array:

    def get_most_common_value(image: np.array, x: int, y: int, kernel_size: int):
        x_kernel_start = x - int(((kernel_size - 1) / 2))
        y_kernel_start = y - int(((kernel_size - 1) / 2))

        values = {}

        for i in range(kernel_size):
            for j in range(kernel_size):
                curr_x = x_kernel_start + i
                curr_y = y_kernel_start + j
                if 0 <= curr_x < image.shape[0] and 0 <= curr_y < image.shape[1]:
                    if image[curr_x][curr_y] not in values:
                        values[image[curr_x][curr_y]] = 1
                    else:
                        values[image[curr_x][curr_y]] = values[image[curr_x][curr_y]] + 1

        max_value = -1
        max_key = None
        for key, value in values.items():
            if value > max_value:
                max_value = value
                max_key = key

        return max_key

    if kernel_size % 2 == 0:
        raise Exception("The size of the kernel must be odd")

    cleaned_image = np.copy(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            cleaned_image[i][j] = get_most_common_value(image, i, j, kernel_size)

    return cleaned_image


def remove_noise_from_labeled_image(image: np.array, labels: typing.Dict) -> np.array:

    def find_closest_value(target, values):
        closest_value = None

        for value in values:
            if closest_value is None or abs(target - closest_value) > abs(target - value):
                closest_value = value

        return closest_value

    labels_values = list(labels.values())

    clean_image = np.copy(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            clean_image[i][j] = find_closest_value(image[i][j], labels_values)

    return clean_image


def get_different_values_image(image: np.array) -> typing.List:
    values = set()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            values.add(image[i][j])

    return list(values)


