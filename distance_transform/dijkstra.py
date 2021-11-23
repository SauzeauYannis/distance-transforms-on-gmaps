import typing
from heapq import heapify, heappush, heappop


def generalized_dijkstra_dt_gmap(gmap, seed_labels: typing.List[int], propagation_labels: typing.List[int],
                                 accumulation_directions: typing.List[bool] = None) -> None:
    """
    It also saves for each dart the connected_component_label of the closest seed.
    It is useful for the generation of voronoi diagrams.

    It does make no sense to use the algorithms also for faces.
    I do that only for test purposes.
    """

    """
    At the moment the heap implementation that I am using doesn't has a method to update a value
    So I add values to the heap even if it is already present.
    In order to not consider the same dart multiple times, I need a visited structure.
    """

    # Initialization
    for dart in gmap.darts:
        gmap.distances[dart] = -1

    # Initialize accumulation directions if None
    if accumulation_directions is None:
        accumulation_directions = []
        for i in range(gmap.n + 1):
            accumulation_directions.append(True)

    # Initialize distance to 0 for seeds and add the seeds to the heap
    heap = []  # each element is a tuple -> (distance, dart)
    heapify(heap)
    for dart in gmap.darts:
        if gmap.image_labels[dart] in seed_labels:
            heappush(heap, (gmap.distances[dart], dart))
            gmap.distances[dart] = 0
            gmap.dt_connected_components_labels[dart] = gmap.connected_components_labels[dart]

    # visited
    visited = set()

    while len(heap) > 0:
        distance, dart = heappop(heap)
        visited.add(dart)
        # Visit all the neighbours
        for i in range(gmap.n + 1):
            neighbour = gmap.ai(i, dart)

            # check if visited
            if neighbour in visited:
                continue

            # Check if I can propagate to that dart
            if gmap.image_labels[neighbour] not in propagation_labels:
                continue

            if accumulation_directions[i]:
                if gmap.distances[neighbour] == -1 \
                        or gmap.distances[dart] + gmap.weights[dart] < gmap.distances[neighbour]:
                    gmap.distances[neighbour] = gmap.distances[dart] + gmap.weights[dart]
                    gmap.dt_connected_components_labels[neighbour] = gmap.dt_connected_components_labels[dart]
                    heappush(heap, (gmap.distances[neighbour], neighbour))
            else:
                if gmap.distances[neighbour] == -1 \
                        or gmap.distances[dart] < gmap.distances[neighbour]:
                    gmap.distances[neighbour] = gmap.distances[dart]
                    gmap.dt_connected_components_labels[neighbour] = gmap.dt_connected_components_labels[dart]
                    heappush(heap, (gmap.distances[neighbour], neighbour))

