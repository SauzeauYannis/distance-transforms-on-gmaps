"""
Evaluate performance compute diffusion distance.

In particular the difference between wave_propagation and Dijkstra are analyzed.
"""

import logging
import time
import typing

import logging_configuration
import cv2
from combinatorial.pixelmap import LabelMap
import tracemalloc
from memory_profiler import profile
from distance_transform.wave_propagation import *
from combinatorial.utils import build_dt_grey_image
from distance_transform.preprocessing import *
from distance_transform.dt_applications import *


# get logger
logger = logging.getLogger("evaluate_performance_compute_diffusion_distance_logger")
logging_configuration.set_logging("results")


def measure_time_build_gmap(image):
    start = time.time()
    gmap = LabelMap.from_labels(image, False)
    end = time.time()
    logger.info(f"gmap with {image.shape[0]*image.shape[1]*8} darts successfully builded in {end-start} seconds")

    return gmap


def measure_memory_build_gmap_tracemalloc(image) -> None:
    shape = image.shape
    tracemalloc.start()
    gmap = LabelMap.from_labels(image, False)
    traced_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info(f"Memory: current: {traced_memory[0]/1000000} MB peak: {traced_memory[1]/1000000} MB")
    logger.info(f"Memory required for each dart is: {traced_memory[0]/(shape[0]*shape[1]*8)} B")


def measure_time_wave_propagation(gmap) -> None:
    # I assume that for the project is required to propagate distances through faces
    accumulation_directions = generate_accumulation_directions_cell(2)
    start = time.time()
    wave_propagation_dt_gmap(gmap, None, accumulation_directions=accumulation_directions)
    end = time.time()
    logger.info(f"wave propagation successfully executed in {end-start} seconds")


def measure_time_edges_reduction(gmap, reduction_factor: float) -> None:
    start = time.time()
    gmap.remove_edges(reduction_factor)
    end = time.time()
    logger.info(f"edges successfully removed in {end-start} seconds. Reduction factor: {reduction_factor}")


def measure_time_vertices_reduction(gmap) -> None:
    start = time.time()
    gmap.remove_vertices()
    end = time.time()
    logger.info(f"vertices successfully removed in {end-start} seconds.")


def measure_time_image_creation(gmap) -> None:
    start = time.time()
    build_dt_grey_image(gmap)
    end = time.time()
    logger.info(f"image successfully created in {end-start} seconds.")


@profile
def measure_memory_build_gmap_memory_profiler(image) -> None:
    shape = image.shape
    gmap = LabelMap.from_labels(image, False)


def read_image(image_path: str) -> None:
    logger.info(f"Reading image from {image_path}")
    image = cv2.imread(image_path, 0)
    logger.info(f"Image shape: {image.shape}")
    return image


def evaluate_performance(image: np.array, out_images_path: str, verbose: bool,
                         compute_voronoi_diagram: bool, reduction_factor: float,
                         use_weights: bool):
    random.seed(42)
    logger.info("")
    logger.info(f"Evaluate performance called")
    logger.info(f"Reduction_factor: {reduction_factor}")
    logger.info(f"Use weights: {use_weights}")
    dt_image_path = out_images_path + "_dt.png"
    gmap, time_to_reduce_gmap_s, time_to_compute_dt_s = compute_dt_for_diffusion_distance(image, dt_image_path, verbose,
                                                                                          compute_voronoi_diagram,
                                                                                          reduction_factor,
                                                                                          use_weights, 50)
    logger.info(f"Time to reduce gmap s: {time_to_reduce_gmap_s}")
    logger.info(f"Time to compute dt s: {time_to_compute_dt_s}")

    start = time.time()
    diffusion_distance = compute_diffusion_distance(gmap, 50)
    end = time.time()
    time_to_compute_diffusion_s = end - start
    logger.info(f"Time to compute diffusion s: {time_to_compute_diffusion_s}")
    logger.info(f"Diffusion distance: {diffusion_distance}")

    if compute_voronoi_diagram:
        voronoi_image_path = out_images_path + "_voronoi.png"
        dt_voronoi_diagram = gmap.generate_dt_voronoi_diagram([labels["stomata"]])
        cv2.imwrite(voronoi_image_path, dt_voronoi_diagram)

    return gmap


def main():
    """
    What to evaluate?
    Respect the diffusion computed on the unreduced gmap, evaluate:
    - absolute error: absolute difference between diffusion computed on the unreduced gmap and
                      diffusion computed on the reduced one
    - relative error: same as absolute error but relative
    - time used to reduce: time used to reduce the gmap
    - time used to compute: time used to compute the gmap

    """

    # For the moment I am using only one image
    image_name = "../data/time_1/cross/DEHYDRATION_small_leaf_4_time_1_ax1cros_0950_Label_1152x1350_uint8.png"
    logger.info(f"Image name: {image_name}")
    image = cv2.imread(image_name, 0)  # the second parameter with value 0 is needed to read the greyscale image
    logger.info(f"Image shape: {image.shape}")
    image_reduction_factor = 11
    logger.info(f"Image reduction factor: {image_reduction_factor}")
    reduced_image = reduce_image_size(image, image_reduction_factor)
    logger.info(f"Image shape after reduction: {reduced_image.shape}")
    cv2.imwrite(f"images/reduced_{image_reduction_factor}_leaf.png", reduced_image)

    evaluate_performance(reduced_image, f"images/reduced_{image_reduction_factor}", False, True, 0, False)
    evaluate_performance(reduced_image, f"images/reduced_{image_reduction_factor}_dijkstra", False, True, 0, True)
    evaluate_performance(reduced_image, f"images/reduced_{image_reduction_factor}_reduced_025", False, True, 0.25, False)
    evaluate_performance(reduced_image, f"images/reduced_{image_reduction_factor}_reduced_dijkstra_025", False, True, 0.25, True)
    evaluate_performance(reduced_image, f"images/reduced_{image_reduction_factor}_reduced_050", False, True, 0.5, False)
    evaluate_performance(reduced_image, f"images/reduced_{image_reduction_factor}_reduced_dijkstra_050", False, True, 0.5, True)
    gmap = evaluate_performance(reduced_image, f"images/reduced_{image_reduction_factor}_reduced_1", False, True, 1, False)
    #gmap.plot(attribute_to_show="weight")
    #gmap.plot_dt()
    gmap = evaluate_performance(reduced_image, f"images/reduced_{image_reduction_factor}_reduced_dijkstra_1", False, True, 1, True)
    #gmap.plot_dt()


if __name__ == "__main__":
    main()
