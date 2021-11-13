import typing

import numpy as np
import math
import random
from queue import Queue

random.seed(42)

def generalized_find_borders(image: np.array, region_label_value: int, border_label_value: int) -> np.array:
    """
    It finds the borders of the regions with label_value and creates a new image identical with
    the image passed parameters but with the borders of the interested regions modified with
    border_label_value

    :param image:
    :param label_value:
    :param border_label_value:
    :return:
    """

    new_image = np.copy(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == region_label_value and not is_equal_to_neighbours(image, (i, j)):
                new_image[i][j] = border_label_value

    return new_image


def is_equal_to_neighbours(image: np.array, value_position: typing.Tuple[int, int]) -> True:
    value_x = value_position[0]
    value_y = value_position[1]
    x_values = [value_x, value_x - 1, value_x, value_x + 1]
    y_values = [value_y -1, value_y, value_y + 1, value_y]
    for x, y in zip(x_values, y_values):
            if 0 < x < image.shape[0] and 0 < y < image.shape[1] and image[x][y] != image[value_x][value_y]:
                return False
    return True


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

    if kernel_size % 2 == 0:
        raise Exception("The size of the kernel must be odd")

    cleaned_image = np.copy(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            cleaned_image[i][j] = get_most_common_value(image, i, j, kernel_size)

    return cleaned_image


def get_most_common_value(image: np.array, x: int, y: int, kernel_size: int, center_kernel: bool = True):
    """
    Center kernel specify if the kernel has to be centered in the pixel value or not.
    If not the current pixel is positioned in the upper left position of the kernel
    """

    if center_kernel:
        x_kernel_start = x - int(((kernel_size - 1) / 2))
        y_kernel_start = y - int(((kernel_size - 1) / 2))
    else:
        x_kernel_start = x
        y_kernel_start = y

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


def reduce_image_size(image: np.array, kernel_size: int) -> np.array:
    """
    It reduces the size of the image passed as parameter.

    The shape of the new image is equal to ceil(shape / kernel size)
    For each pixel a kernel of size "kernel_size" is considered.
    The considered pixel is located in the upper left corner of the kernel.
    The value of the new pixel is the most common value in the image inside the kernel.

    Example:
    kernel_size = 2

    1 1 3       1 3
    0 1 3  ->   0 2
    0 0 2

    Explanation:
    1 1 -> 1   3 -> 3
    0 1        3

    0 0 -> 0   2 -> 2

    :param image:
    :param kernel_size:
    :return:
    """

    new_shape = [math.ceil(shape / kernel_size) for shape in image.shape]
    reduced_image = np.zeros(new_shape)

    for i in range(reduced_image.shape[0]):
        for j in range(reduced_image.shape[1]):
            reduced_image[i][j] = get_most_common_value(image, i * kernel_size, j * kernel_size, kernel_size)

    return reduced_image


def connected_component_labeling_one_pass(image: np.array) -> np.array:
    """
    Pay attention to the number of connected components.
    It influences the size of the resulting numpy array.
    """

    """
    The maximum number of connected components is equal to the number of pixels in the worst case
    I assume that 4 bytes will be sufficient.
    """
    labeled_image = np.zeros(image.shape, dtype=np.int32)

    # initialization
    # Temporary initialize to -1 if no label has been assigned
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            labeled_image[i][j] = -1

    next_label = 0

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if labeled_image[i][j] == -1:
                # A new connected component has been found
                # Assign the same label to the connected component using wave propagation algorithm
                wave_propagation_labeling(image, labeled_image, (i, j), next_label)
                next_label += 1

    return labeled_image


def wave_propagation_labeling(image: np.array, labeled_image: np.array,
                              position: typing.Tuple[int, int], label: int) -> None:
    """
    4 connected neighborhood
    """
    def is_point_in_connected_component(image: np.array, labeled_image: np.array,
                                        position: typing.Tuple[int, int], label: int) -> bool:
        # check if point is in image
        if position[0] < 0 or position[0] >= image.shape[0] or position[1] < 0 or position[1] >= image.shape[1]:
            return False

        # check if the point has already got a label
        if labeled_image[position[0]][position[1]] != -1:
            return False

        # check if the point has the same label of the connected component in the image
        if image[position[0]][position[1]] != label:
            return False

        return True

    queue = Queue()
    labeled_image[position[0]][position[1]] = label
    queue.put(position)

    while not queue.empty():
        curr_position = queue.get()

        # check neighbours
        # up
        next_position = (curr_position[0] - 1, curr_position[1])
        if is_point_in_connected_component(image, labeled_image, next_position,
                                           image[curr_position[0]][curr_position[1]]):
            labeled_image[next_position[0]][next_position[1]] = label
            queue.put(next_position)

        # right
        next_position = (curr_position[0], curr_position[1] + 1)
        if is_point_in_connected_component(image, labeled_image, next_position,
                                           image[curr_position[0]][curr_position[1]]):
            labeled_image[next_position[0]][next_position[1]] = label
            queue.put(next_position)

        # down
        next_position = (curr_position[0] + 1, curr_position[1])
        if is_point_in_connected_component(image, labeled_image, next_position,
                                           image[curr_position[0]][curr_position[1]]):
            labeled_image[next_position[0]][next_position[1]] = label
            queue.put(next_position)

        # left
        next_position = (curr_position[0], curr_position[1] - 1)
        if is_point_in_connected_component(image, labeled_image, next_position,
                                           image[curr_position[0]][curr_position[1]]):
            labeled_image[next_position[0]][next_position[1]] = label
            queue.put(next_position)


def build_rgb_image_from_labeled_image(labeled_image: np.array) -> np.array:

    def generate_random_color() -> typing.Tuple[np.uint8, np.uint8, np.uint8]:
        r = math.floor(random.random() * 255)
        g = math.floor(random.random() * 255)
        b = math.floor(random.random() * 255)

        return r, g, b

    rgb_image_shape = (labeled_image.shape[0], labeled_image.shape[1], 3)
    rgb_image = np.zeros(rgb_image_shape, dtype=np.uint8)

    colors = {}

    for i in range(labeled_image.shape[0]):
        for j in range(labeled_image.shape[1]):
            if labeled_image[i][j] not in colors:
                colors[labeled_image[i][j]] = generate_random_color()

            rgb_image[i][j] = colors[labeled_image[i][j]]

    return rgb_image
