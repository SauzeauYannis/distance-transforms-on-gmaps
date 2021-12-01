import typing
from queue import Queue
import numpy as np


def generalized_wave_propagation_image(image: np.array, seed_labels: typing.List[int],
                                       propagation_labels: typing.List[int]) -> np.array:
    # int64 should be sufficient
    output_image = np.zeros(image.shape, np.int64)
    output_image.fill(-1)  # initialize output_image

    queue = Queue()

    # Find seeds and add to queue
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] in seed_labels:
                queue.put((i, j))
                output_image[i][j] = 0

    while not queue.empty():
        pixel = queue.get()
        # Visit all the neighbours
        for i in range(4):
            neighbour = get_next_neighbour_image(i, pixel[0], pixel[1], image.shape[0] - 1, image.shape[1] - 1)
            if neighbour is not None and image[neighbour[0]][neighbour[1]] in propagation_labels and\
                    output_image[neighbour[0], neighbour[1]] == -1:
                output_image[neighbour[0], neighbour[1]] = output_image[pixel[0], pixel[1]] + 1
                queue.put(neighbour)

    return output_image
    

def wave_propagation_dt_image(image: np.array, seeds: typing.List[typing.Tuple[int, int]] = None) -> np.array:
    """
    4-neighborhood connection

    :param image:
    :param seeds: A list of seeds (x, y). If None, all values equal to 0 in the image will be used as seeds
    :return:
    """

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
            neighbour = get_next_neighbour_image(i, pixel[0], pixel[1], image.shape[0] - 1, image.shape[1] - 1)
            if neighbour is not None and output_image[neighbour[0], neighbour[1]] == -1:
                output_image[neighbour[0], neighbour[1]] = output_image[pixel[0], pixel[1]] + 1
                queue.put(neighbour)

    return output_image


def get_next_neighbour_image(index: int, x: int, y: int, max_x: int, max_y: int) -> (int, int):
    """
    4 neighborhood connections

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


def improved_wave_propagation_gmap_vertex(gmap, seed_labels: typing.List[int], propagation_labels: typing.List[int]) -> None:
    """
    It computes only dt, without saving information for the generation of voronoi diagram

    I add to the queue only one dart per vertex
    """

    def set_distance_if_in_propagation_and_seed_label(gmap, dart, distance):
        gmap.distances[dart] = distance
        """
        if gmap.image_labels[dart] in propagation_labels:
            gmap.distances[dart] = distance
        """


    def get_neighbours_vertices(gmap, dart):
        yield gmap.a0(dart)
        yield gmap.a0(gmap.a1(dart))
        yield gmap.a0(gmap.a1(gmap.a2(dart)))
        yield gmap.a0(gmap.a1(gmap.a2(gmap.a1(dart))))

    def set_distance_vertex(gmap, dart, distance):
        gmap.distances[dart] = distance
        set_distance_if_in_propagation_and_seed_label(gmap, gmap.a1(dart), distance)
        set_distance_if_in_propagation_and_seed_label(gmap, gmap.a2(dart), distance)
        set_distance_if_in_propagation_and_seed_label(gmap, gmap.a2(gmap.a1(dart)), distance)
        set_distance_if_in_propagation_and_seed_label(gmap, gmap.a1(gmap.a2(dart)), distance)
        set_distance_if_in_propagation_and_seed_label(gmap, gmap.a2(gmap.a1(gmap.a2(dart))), distance)
        set_distance_if_in_propagation_and_seed_label(gmap, gmap.a1(gmap.a2(gmap.a1(dart))), distance)
        set_distance_if_in_propagation_and_seed_label(gmap, gmap.a2(gmap.a1(gmap.a2(gmap.a1(dart)))), distance)

    # Initialization
    for dart in gmap.darts:
        gmap.distances[dart] = -1

    queue = Queue()
    for dart in gmap.darts:
        if gmap.image_labels[dart] in seed_labels:
            queue.put(dart)
            set_distance_vertex(gmap, dart, 0)

    while not queue.empty():
        dart = queue.get()
        # Visit all the neighbouring vertices
        for neighbour in get_neighbours_vertices(gmap, dart):
            # Check if I can propagate to that dart
            if gmap.image_labels[neighbour] not in propagation_labels:
                continue

            if gmap.distances[neighbour] == -1:
                # put to the same distance to all the darts of the vertex
                set_distance_vertex(gmap, neighbour, gmap.distances[dart] + 1)
                queue.put(neighbour)


def generalized_wave_propagation_gmap(gmap, seed_labels: typing.List[int], propagation_labels: typing.List[int],
                                      accumulation_directions: typing.List[bool] = None) -> None:
    """
    It also saves for each dart the connected_component_label of the closest seed.
    It is useful for the generation of voronoi diagrams.
    """

    # Initialization
    for dart in gmap.darts:
        gmap.distances[dart] = -1

    # Initialize accumulation directions if None
    if accumulation_directions is None:
        accumulation_directions = []
        for i in range(gmap.n + 1):
            accumulation_directions.append(True)

    # Initialize distance to 0 for seeds and add the seeds to the queue
    curr_queue = Queue()
    next_queue = Queue()
    for dart in gmap.darts:
        if gmap.image_labels[dart] in seed_labels:
            curr_queue.put(dart)
            gmap.distances[dart] = 0
            gmap.dt_connected_components_labels[dart] = gmap.connected_components_labels[dart]

    while not curr_queue.empty():
        while not curr_queue.empty():
            dart = curr_queue.get()
            # Visit all the neighbours
            for i in range(gmap.n + 1):
                neighbour = gmap.ai(i, dart)
                # Check if I can propagate to that dart
                if gmap.image_labels[neighbour] not in propagation_labels:
                    continue

                # Due to the accumulation policies, it happens that the first distance value associated to a neighbour
                # could not be the right one. All the values have to be checked. View example on joplin (01/11/2021)
                # I don't remember exactly what this part do. I have to check again.
                if accumulation_directions[i]:
                    if gmap.distances[neighbour] == -1:
                        next_queue.put(neighbour)
                    if gmap.distances[neighbour] == -1 \
                            or gmap.distances[dart] + 1 < gmap.distances[neighbour]:
                        gmap.distances[neighbour] = gmap.distances[dart] + 1
                        gmap.dt_connected_components_labels[neighbour] = gmap.dt_connected_components_labels[dart]
                else:
                    if gmap.distances[neighbour] == -1:
                        curr_queue.put(neighbour)
                    if gmap.distances[neighbour] == -1 \
                            or gmap.distances[dart] < gmap.distances[neighbour]:
                        gmap.distances[neighbour] = gmap.distances[dart]
                        gmap.dt_connected_components_labels[neighbour] = gmap.dt_connected_components_labels[dart]
        curr_queue = next_queue
        next_queue = Queue()


def wave_propagation_dt_gmap(gmap, seeds_identifiers: typing.Optional[typing.List[int]], accumulation_directions: typing.List[bool] = None) -> None:
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

    seed_identifiers: if None all the darts with label equal to 0 will be used and the distance
                      propagates only in the non foreground darts (!= 255)
    """

    # Initialization
    for dart in gmap.darts:
        gmap.distances[dart] = -1

    # Initialize accumulation directions if None
    if accumulation_directions is None:
        accumulation_directions = []
        for i in range(gmap.n + 1):
            accumulation_directions.append(True)

    curr_queue = Queue()
    next_queue = Queue()
    if seeds_identifiers is None:
        for dart in gmap.darts:
            if gmap.image_labels[dart] == 0:
                curr_queue.put(dart)
                gmap.distances[dart] = 0
    else:
        for seed_identifier in seeds_identifiers:
            curr_queue.put(seed_identifier)
            gmap.distances[seed_identifier] = 0

    while not curr_queue.empty():
        while not curr_queue.empty():
            dart = curr_queue.get()
            # Visit all the neighbours
            for i in range(gmap.n + 1):
                neighbour = gmap.ai(i, dart)
                if seeds_identifiers is None and gmap.image_labels[neighbour] == 255:
                    continue
                # due to the accumulation policies, it happens that the first distance value associated to a neighbours
                # could not be the right one. All the values have to be checked. View example on joplin (01/11/2021)
                if accumulation_directions[i]:
                    if gmap.distances[neighbour] == -1:
                        next_queue.put(neighbour)  # only the first time I add the element to the queue
                    if gmap.distances[neighbour] == -1\
                            or gmap.distances[dart] + 1 < gmap.distances[neighbour]:
                        gmap.distances[neighbour] = gmap.distances[dart] + 1
                else:
                    if gmap.distances[neighbour] == -1:
                        curr_queue.put(neighbour)  # only the first time I add the element to the queue
                    if gmap.distances[neighbour] == -1 \
                                or gmap.distances[dart] < gmap.distances[neighbour]:
                        gmap.distances[neighbour] = gmap.distances[dart]
        curr_queue = next_queue
        next_queue = Queue()


def generate_accumulation_directions_vertex(gmap_size: int) -> typing.List[bool]:
    accumulation_directions = []

    accumulation_directions.append(True)
    for i in range(gmap_size):
        accumulation_directions.append(False)

    return accumulation_directions


def generate_accumulation_directions_cell(gmap_size: int) -> typing.List[bool]:
    """
    Generates the accumulation directions array for the cell of the max dimension.
    For a 2gmap is the face (2cell) for a 3gmap is the volume (3cell).
    Note that the face in a 2gmap corresponds to a pixel in the corresponding image.
    and the volume in a 3gmap corresponds to a voxel in the corresponding 3d image.

    Note:
    The size of a n-gmap is n.
    So for a 2gmap the gmap size is equal to 2.

    """
    accumulation_directions = []

    for i in range(gmap_size):
        accumulation_directions.append(False)
    accumulation_directions.append(True)

    return accumulation_directions

