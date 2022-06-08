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
from distance_transform.wave_propagation import generalized_wave_propagation_gmap, generalized_wave_propagation_image, generate_accumulation_directions_vertex


def compute_diffusion_distance(gmap, label: int) -> float:
    """
    It's the average of the distance of each point with label equal to the label passed as parameter

    At the moment I am computing the average on darts
    Probably I should compute the average on faces. It is not so difficult to do that

    I consider only the darts with distance value >= 0.
    Read Joplin 25/11/2021 - Bug compute diffusion distance

    It returns -1 if no values have been found.
    It could happen if there are no stomata in the image.
    """

    sum = 0
    number_of_values = 0

    for dart in gmap.darts:
        if gmap.image_labels[dart] == label and gmap.distances[dart] >= 0:
            number_of_values += 1
            sum += gmap.distances[dart]

    if number_of_values == 0:
        return -1

    average = sum / number_of_values
    return average


def compute_diffusion_distance_image(image: np.array, dt_image: np.array, label: int) -> float:
    """
    It's the average of the distance of each point with label equal to the label passed as parameter.

    I consider only the pixels with distance value >= 0.
    Read Joplin 25/11/2021 - Bug compute diffusion distance

    It returns -1 if no values have been found.
    It could happen if there are no stomata in the image.
    """

    sum = 0
    number_of_values = 0

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == label and dt_image[i][j] >= 0:
                number_of_values += 1
                sum += dt_image[i][j]

    if number_of_values == 0:
        return -1

    average = sum / number_of_values
    return average


def compute_dt_for_diffusion_distance(image: np.array, dt_image_path: str = None, verbose: bool = False,
                                      compute_voronoi_diagram: bool = False, reduction_factor: float = 0,
                                      use_weights: bool = False) -> typing.Tuple[nGmap, float, float]:
    """
    Computes the diffusion distance of the cell represented by the image in image_path.

    Add the parameter save_gmap in order to save the gmap
    Add the parameter reduction_factor in order to compute dt on the reduced gfap

    It returns a tuple of t
    The first element is the time, in seconds, required to reduce the gmap
    The second element is the time, in seconds, required to compure dt
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
        print("Gmap successfully builded")
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
                                                          labels["stomata"], labels['air']], interpolate_missing_values=True)
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

    # Decomment to generate contour plot
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
    gmap = LabelMap.from_labels(image)
    generalized_wave_propagation_gmap(gmap, [labels["stomata"]], [
                                      labels['air']], [labels['cell']])

    return gmap.distances.reshape(image.shape[0], image.shape[1], 8)


def compute_dt_inside_air(image: np.array) -> np.array:
    # Identiy the air region reacheable by the stomata
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
    image_with_borders = generalized_find_borders(image, labels["air"], 180)
    gmap = LabelMap.from_labels(image_with_borders)
    generalized_wave_propagation_gmap(
        gmap, [labels["stomata"]], [180], [labels["cell"]])

    return gmap.distances.reshape(image.shape[0], image.shape[1], 8)
