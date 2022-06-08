import os
import time
import typing

import cv2
import numpy as np

from combinatorial.pixelmap import LabelMap
from combinatorial.gmaps import nGmap
from combinatorial.utils import build_dt_grey_image_from_gmap
from data.labels import labels
from distance_transform.dijkstra import generalized_dijkstra_dt_gmap
from distance_transform.preprocessing import generalized_find_borders, connected_component_labeling_one_pass
from distance_transform.wave_propagation import generalized_wave_propagation_gmap, generalized_wave_propagation_image,\
    generate_accumulation_directions_vertex


def compute_diffusion_distance(gmap, label: int) -> float:
    """It's the average of the distance of each point with label equal to the label passed as parameter

    At the moment I am computing the average on darts
    Probably I should compute the average on faces. It is not so difficult to do that

    I consider only the darts with distance value >= 0.
    Read Joplin 25/11/2021 - Bug compute diffusion distance

    Args:
        gmap: the gmap of the cell.
        label: the label of the cell.

    Returns:
        float: the average diffusion distance of the label.
        -1 if the label is not present in the gmap.
    """

    distance_sum = 0
    number_of_values = 0

    for dart in gmap.darts:
        if gmap.image_labels[dart] == label and gmap.distances[dart] >= 0:
            number_of_values += 1
            distance_sum += gmap.distances[dart]

    if number_of_values == 0:
        return -1

    return distance_sum / number_of_values


def compute_diffusion_distance_image(image: np.array, dt_image: np.array, label: int) -> float:
    """Same as compute_diffusion_distance but for an image.

    Args:
        image: the image.
        dt_image: the diffusion distance image.
        label: the label of the cell.

    Returns:
        float: the average diffusion distance of the label.
        -1 if the label is not present in the image.
    """

    distance_sum = 0
    number_of_values = 0

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == label and dt_image[i][j] >= 0:
                number_of_values += 1
                distance_sum += dt_image[i][j]

    if number_of_values == 0:
        return -1

    return distance_sum / number_of_values


def compute_dt_for_diffusion_distance(
    image: np.array,
    dt_image_path: str = None,
    verbose: bool = False,
    compute_voronoi_diagram: bool = False,
    reduction_factor: float = 0,
    use_weights: bool = False
) -> typing.Tuple[nGmap, float, float]:
    """Computes the diffusion distance of the cell represented by the image in image_path.

    Args:
        image: the image of the cell.
        dt_image_path: the path to the diffusion distance image.
        verbose: if True, it prints the different steps of the computation.
        compute_voronoi_diagram: if True, it computes the voronoi diagram of the cell.
        reduction_factor: the reduction factor of the image.
        use_weights: if True, it uses the weights of the gmap.

    Returns:
        typing.Tuple[nGmap, float, float]: the gmap of the cell, the time required to reduce the gmap
        and the time required to compute the distance transform.
    """

    connected_components_labels = None
    if compute_voronoi_diagram:
        # Find connected components labels
        connected_components_labels = connected_component_labeling_one_pass(
            image)

    # Build gmap
    gmap = LabelMap.from_labels(
        image, connected_components_labels=connected_components_labels)
    if verbose:
        print("Gmap successfully build")
    # gmap.uniform_labels_for_vertices()  # used if the improvement algorithm is used. Does not work good
        # I can't do that.

    # Reduce gmap
    start = time.time_ns()
    if reduction_factor > 0:
        gmap.remove_edges(reduction_factor)
        gmap.remove_vertices()
        if verbose:
            print(
                f"Gmap successfully reduced with reduction factor: {reduction_factor}")
    end = time.time_ns()
    time_to_reduce_gmap_s = (end - start) / (10 ** 9)

    # Compute dt from stomata to the cells
    accumulation_directions = generate_accumulation_directions_vertex(2)
    start = time.time_ns()
    if use_weights:
        generalized_dijkstra_dt_gmap(gmap, [labels["stomata"]], [labels['air']], [
                                     labels['cell']], accumulation_directions)
    else:
        generalized_wave_propagation_gmap(gmap, [labels["stomata"]], [labels['air']], [
                                          labels['cell']], accumulation_directions)
        # improved_wave_propagation_gmap_vertex(gmap, [labels["stomata"]], propagation_labels)
    end = time.time_ns()
    time_to_compute_dt_s = (end - start) / (10 ** 9)
    if verbose:
        print("Dt successfully computed")

    # Save dt image
    dt_image_real_distances = gmap.build_dt_image(propagation_labels=[
                                                  labels["stomata"], labels['air']], interpolate_missing_values=True)
    dt_image = build_dt_grey_image_from_gmap(gmap, propagation_labels=[
                                             labels["stomata"], labels['air']], interpolate_missing_values=False)
    dt_image_interpolated = build_dt_grey_image_from_gmap(gmap, propagation_labels=[
                                                          labels["stomata"], labels['air']],
                                                          interpolate_missing_values=True)
    if verbose:
        print("dt image successfully computed")

    if dt_image_path:
        real_distances_image_name = os.path.splitext(
            dt_image_path)[0] + "_real_distances.npy"
        np.save(real_distances_image_name, dt_image_real_distances)
        cv2.imwrite(dt_image_path, dt_image)
        interpolated_image_name = os.path.splitext(
            dt_image_path)[0] + "_interpolated.png"
        cv2.imwrite(interpolated_image_name, dt_image_interpolated)

    # Uncomment to generate contour plot
    # contour_plot_from_dt_image(dt_image_real_distances, 20, 400)

    return gmap, time_to_reduce_gmap_s, time_to_compute_dt_s


def compute_dt_for_diffusion_distance_image(image: np.array) -> typing.Tuple[np.array, float]:
    """Computes the diffusion distance of the cell represented by the image in image_path.

    Args:
        image: the image of the cell.

    Returns:
        typing.Tuple[np.array, float]: The diffusion distance image and the time required to compute it.
    """

    # Compute dt from stomata to the cells
    start = time.time_ns()
    dt_image = generalized_wave_propagation_image(
        image, [labels["stomata"]], [labels['air']], [labels['cell']])
    end = time.time_ns()
    time_to_compute_dt_s = (end - start) / (10 ** 9)

    return dt_image, time_to_compute_dt_s


def compute_dt_from_stomata_to_cells(image: np.array) -> np.array:
    """Computes the diffusion distance of the image from stomata to cells propagating in the air.

    Args:
        image: the image of the leaf.

    Returns:
        np.array: the diffusion distance array stored like a gmap.
    """

    gmap = LabelMap.from_labels(image)
    generalized_wave_propagation_gmap(gmap, [labels["stomata"]], [
                                      labels['air']], [labels['cell']])

    return gmap.distances.reshape(image.shape[0], image.shape[1], 8)


def compute_dt_inside_air(image: np.array) -> np.array:
    """Computes the diffusion distance of the image inside the air.

    Args:
        image: the image of the leaf.

    Returns:
        np.array: the diffusion distance array stored like a gmap.
    """

    # Identify the air region reachable by the stomata
    dt_image = generalized_wave_propagation_image(
        image, [labels["stomata"]], [labels['air']], [labels['air']])
    air_image = np.copy(dt_image)  # to modify, I have to remove stomata
    for i in range(air_image.shape[0]):
        for j in range(air_image.shape[1]):
            if air_image[i][j] == -1 or image[i][j] != labels["air"]:
                air_image[i][j] = 0
            else:
                air_image[i][j] = labels["air"]

    image_with_borders = generalized_find_borders(
        air_image, labels["air"], 180)
    gmap = LabelMap.from_labels(image_with_borders)
    generalized_wave_propagation_gmap(
        gmap, [180], [labels['air']], [labels['air']])

    return gmap.distances.reshape(image.shape[0], image.shape[1], 8)


def compute_dt_from_stomata_to_cells_using_borders(image: np.array) -> np.array:
    """Computes the diffusion distance from stomata to cells using the borders of the air region.

    Args:
        image: the image of the leaf.

    Returns:
        np.array: the diffusion distance array stored like a gmap.
    """

    image_with_borders = generalized_find_borders(image, labels["air"], 180)
    gmap = LabelMap.from_labels(image_with_borders)
    generalized_wave_propagation_gmap(
        gmap, [labels["stomata"]], [180], [labels["cell"]])

    return gmap.distances.reshape(image.shape[0], image.shape[1], 8)
