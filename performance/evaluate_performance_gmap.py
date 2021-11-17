"""
Evaluate performance gmap
"""

import logging
import time

import logging_configuration
import cv2
from combinatorial.pixelmap import LabelMap
import tracemalloc
from memory_profiler import profile
from distance_transform.wave_propagation import *

# temp
# from combinatorial_original.pixelmap import LabelMap

# get logger
logger = logging.getLogger("evaluate_performance_gmap_logger")
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
    gmap.build_dt_color_image()
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


def evaluate_performance(image_path: str, reduction_fraction: float) -> None:
    image = read_image(image_path)

    # I have to measure time and memory separately
    # Because the measurement of the memory slows down all the operations

    # build gmap
    # wave propagation
    # edges reduction (specify fraction)
    # vertices reduction (specify fraction)
    # build image after reduction
    # wave propagation again?

    # time measurement
    # build
    gmap = measure_time_build_gmap(image)
    measure_time_wave_propagation(gmap)
    measure_time_edges_reduction(gmap, reduction_fraction)
    measure_time_vertices_reduction(gmap)
    measure_time_image_creation(gmap)

    measure_memory_build_gmap_tracemalloc(image)



def main():
    # 5.6 GB estimate for 1000x1000 image

    evaluate_performance('../data/100_100_portion_leaf.png', reduction_fraction=1)
    evaluate_performance('../data/200_200_portion_leaf.png', reduction_fraction=1)
    evaluate_performance('../data/300_300_portion_leaf.png', reduction_fraction=1)
    # evaluate_performance('../data/1000_1000_portion_leaf.png', reduction_fraction=1)


if __name__ == "__main__":
    main()
