import typing
from combinatorial.gmaps import Dart
from queue import Queue


def wave_propagation_dt(gmap, seeds_identifiers: typing.List[int]) -> None:
    """
    It computes the dt for the gmap passed as parameter.
    The distance propagates through all the cells (using all the involutions).

    A parameter can be added in order to consider only a subset of involutions.

    :param gmap:
    :param seeds_identifiers:
    :return:
    """

    # Initialization
    for dart in gmap.darts_with_attributes:
        dart.attributes["distance"] = None

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
                neighbour.attributes["distance"] = dart.attributes["distance"] + 1
                queue.put(neighbour)
