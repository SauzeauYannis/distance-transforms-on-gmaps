import typing
from combinatorial.gmaps import Dart
from queue import Queue
import numpy as np


def wave_propagation_dt_image(image: np.array, seeds: typing.List[typing.Tuple[int, int]] = None) -> np.array:
    """
    4-neighborhood connection

    :param image:
    :param seeds: A list of seeds (x, y). If None, all values equal to 0 in the image will be used as seeds
    :return:
    """

    def get_next_neighbour(index: int, x: int, y: int, max_x: int, max_y: int) -> (int, int):
        """
        Preconditions:
            x and y are valid values, i.e. x, y >= 0 and x <= max_x and y <= max_y

        :param index:
        :param x:
        :param y:
        :return:
        """
        if index == 0:  # left
            if y - 1 < 0:
                return None
            else:
                return x, y - 1
        elif index == 1:  # up
            if x - 1 < 0:
                return None
            else:
                return x - 1, y
        elif index == 2:  # right
            if y + 1 > max_y:
                return None
            else:
                return x, y + 1
        elif index == 3:  # down
            if x + 1 > max_x:
                return None
            else:
                return x + 1, y
        else:
            raise Exception(f"Unexpected index {index}")

    output_image = np.zeros(image.shape, image.dtype)
    # initialize output_image
    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            output_image[i][j] = -1

    queue = Queue()
    if seeds is None:
        # find seeds and add to queue
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] == 0:
                    queue.put((i, j))
                    output_image[i][j] = 0
    else:
        for seed in seeds:
            output_image[seed[0]][seed[1]] = image[seed[0]][seed[1]]
            queue.put(seed)

    while not queue.empty():
        pixel = queue.get()
        # Visit all the neighbours
        for i in range(4):
            neighbour = get_next_neighbour(i, pixel[0], pixel[1], image.shape[0] - 1, image.shape[1] - 1)
            if neighbour is not None and output_image[neighbour[0], neighbour[1]] == -1:
                output_image[neighbour[0], neighbour[1]] = output_image[pixel[0], pixel[1]] + 1
                queue.put(neighbour)

    return output_image


def wave_propagation_dt_gmap(gmap, seeds_identifiers: typing.List[int], accumulation_directions: typing.List[bool] = None) -> None:
    """
    It computes the dt for the gmap passed as parameter.
    The distance propagates through all the cells (using all the involutions).

    Despite the fact that distance propagates through all the directions, a distance unit is not necessarily added
    for all the directions.
    The accumulation_directions parameter can be used to specify to which directions increase the distance.
    accumulation_directions has to be a list of n+1 elements, where n is the level of the gmap.
    Each element can be True, if the distance has to be increased, or False otherwise.
    Passing None as parameter has the same result of passing an array of False.
    The default behaviour is to increase the distance for all the directions, but a different combination can be used
    to increase distances in a different way.
    For example if the distance has to be increased only passing from a vertex to another, the combination of
    alfa0 = True and alfai = False for all the remaining i
    So the array should be: [True False False] for a 2-gmap
    can be used.
    """

    # Initialization
    for dart in gmap.darts_with_attributes:
        dart.attributes["distance"] = None

    # Initialize accumulation directions if None
    if accumulation_directions is None:
        accumulation_directions = []
        for i in range(gmap.n + 1):
            accumulation_directions.append(True)

    queue = Queue()
    for seed_identifier in seeds_identifiers:
        dart = gmap.get_dart_by_identifier(seed_identifier)
        queue.put(dart)
        dart.attributes["distance"] = 0

    while not queue.empty():
        dart = queue.get()
        # Visit all the neighbours
        for i in range(gmap.n + 1):
            neighbour = gmap.alfa_i(i, dart.identifier)
            if neighbour.attributes["distance"] is None:
                if accumulation_directions[i]:
                    neighbour.attributes["distance"] = dart.attributes["distance"] + 1
                else:
                    neighbour.attributes["distance"] = dart.attributes["distance"]
                queue.put(neighbour)


def generate_accumulation_directions_vertex(gmap_size: int) -> typing.List[int]:
    accumulation_directions = []

    accumulation_directions.append(True)
    for i in range(gmap_size):
        accumulation_directions.append(False)

    return accumulation_directions


def generate_accumulation_directions_cell(gmap_size: int) -> typing.List[int]:
    """
    Generates the accumulation directions array for the cell of the max dimension.
    For a 2gmap is the face (2cell) for a 3gmap is the volume (3cell).
    Note that the face in a 2gmap corresponds to a pixel in the corresponding image.
    and the volume in a 3gmap corresponds to a voxel in the corresponding 3d image.

    """
    accumulation_directions = []

    for i in range(gmap_size):
        accumulation_directions.append(False)
    accumulation_directions.append(True)

    return accumulation_directions

