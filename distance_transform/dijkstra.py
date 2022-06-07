from heapq import heapify, heappush, heappop

import typing


def generalized_dijkstra_dt_gmap(
    gmap,
    seed_labels: typing.List[int],
    propagation_labels: typing.List[int],
    target_labels: typing.List[int],
    accumulation_directions: typing.List[bool] = None
) -> None:
    """Compute the distance transform of a gmap using the Dijkstra algorithm

    Args:
        gmap: the gmap to compute the distance transform
        seed_labels: the labels of the seed points
        propagation_labels: the labels of the propagation points
        target_labels: the labels of the target points
        accumulation_directions: the accumulation directions of the propagation points
    """

    admissible_labels = propagation_labels + target_labels

    # Initialization
    gmap.distances.fill(-1)

    # Initialize accumulation directions if None
    if accumulation_directions is None:
        accumulation_directions = []
        for i in range(gmap.n + 1):
            accumulation_directions.append(True)

    # Initialize distance to 0 for seeds and add the seeds to the heap
    heap = []
    heapify(heap)
    for dart in gmap.darts:
        if gmap.image_labels[dart] in seed_labels:
            gmap.distances[dart] = 0
            heappush(heap, dart)
            gmap.dt_connected_components_labels[dart] = gmap.connected_components_labels[dart]

    # visited
    visited = set()

    while len(heap) > 0:
        dart = heappop(heap)
        visited.add(dart)
        # Visit all the neighbours
        for i in range(gmap.n + 1):
            neighbour = gmap.ai(i, dart)

            # check if visited
            if neighbour in visited:
                continue

            # Check if I can propagate to that dart
            if gmap.image_labels[neighbour] not in admissible_labels:
                continue

            if accumulation_directions[i]:
                if gmap.distances[neighbour] == -1 \
                        or gmap.distances[dart] + gmap.weights[dart] < gmap.distances[neighbour]:
                    gmap.distances[neighbour] = gmap.distances[dart] + \
                        gmap.weights[dart]
                    gmap.dt_connected_components_labels[neighbour] = gmap.dt_connected_components_labels[dart]
                    if gmap.image_labels[neighbour] in propagation_labels:
                        heappush(heap, neighbour)
            else:
                if gmap.distances[neighbour] == -1 \
                        or gmap.distances[dart] < gmap.distances[neighbour]:
                    gmap.distances[neighbour] = gmap.distances[dart]
                    gmap.dt_connected_components_labels[neighbour] = gmap.dt_connected_components_labels[dart]
                    if gmap.image_labels[neighbour] in propagation_labels:
                        heappush(heap, neighbour)
