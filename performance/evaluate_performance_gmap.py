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

# temp
# from combinatorial_original.pixelmap import LabelMap

# get logger
logger = logging.getLogger("evaluate_performance_gmap_logger")
logging_configuration.set_logging("results")


def measure_time_build_gmap(image) -> None:
    start = time.time()
    gmap = LabelMap.from_labels(image, False)
    end = time.time()
    logger.info(f"gmap successfully builded in {end-start} seconds")


def measure_memory_build_gmap_tracemalloc(image) -> None:
    shape = image.shape
    tracemalloc.start()
    gmap = LabelMap.from_labels(image, False)
    traced_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info(f"Memory: current: {traced_memory[0]/1000000} MB peak: {traced_memory[1]/1000000} MB")
    logger.info(f"Memory required for each dart is: {traced_memory[0]/(shape[0]*shape[1]*8)} B")


@profile
def measure_memory_build_gmap_memory_profiler(image) -> None:
    shape = image.shape
    gmap = LabelMap.from_labels(image, False)


def evaluate_performance_build_gmap(path: str) -> None:
    logger.info(f"Reading image from {path}")
    image = cv2.imread(path, 0)
    logger.info(f"Image shape: {image.shape}")

    # I have to measure time and memory separately
    # Because the measurement of the memory slows down the building process
    measure_time_build_gmap(image)
    measure_memory_build_gmap_tracemalloc(image)

def main():
    # 5.6 GB estimate for 1000x1000 image

    evaluate_performance_build_gmap('../data/100_100_portion_leaf.png')
    evaluate_performance_build_gmap('../data/200_200_portion_leaf.png')
    evaluate_performance_build_gmap('../data/300_300_portion_leaf.png')

    evaluate_performance_build_gmap('../data/300_300_portion_leaf.png')

if __name__ == "__main__":
    main()
