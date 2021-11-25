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
import os


# get logger
logger = logging.getLogger("evaluate_performance_compute_diffusion_distance_logger")
logging_configuration.set_logging("results")


def evaluate_performance(image: np.array, out_images_path: typing.Optional[str], verbose: bool,
                         compute_voronoi_diagram: bool, reduction_factor: float,
                         use_weights: bool):
    random.seed(42)

    if out_images_path:
        dt_image_path = out_images_path + "_dt.png"
    else:
        dt_image_path = None
    gmap, time_to_reduce_gmap_s, time_to_compute_dt_s = compute_dt_for_diffusion_distance(image, dt_image_path, verbose,
                                                                                          compute_voronoi_diagram,
                                                                                          reduction_factor,
                                                                                          use_weights, 50)

    start = time.time()
    diffusion_distance = compute_diffusion_distance(gmap, 50)
    end = time.time()
    time_to_compute_diffusion_s = end - start

    if compute_voronoi_diagram and out_images_path:
        voronoi_image_path = out_images_path + "_voronoi.png"
        dt_voronoi_diagram = gmap.generate_dt_voronoi_diagram([labels["stomata"]])
        cv2.imwrite(voronoi_image_path, dt_voronoi_diagram)

    if verbose:
        # print report
        logger.info("{: <20} {: <20} {: <20} {: <20} {: <20} {: <20}".format(f"{reduction_factor}", f"{use_weights}",
                                                                             f"{diffusion_distance}", f"{time_to_reduce_gmap_s}",
                                                                             f"{time_to_compute_dt_s}", f"{time_to_compute_diffusion_s}"))

    return gmap


def evaluate_performance_one_image(image_path) -> None:
    """
    What to evaluate?
    Respect the diffusion computed on the unreduced gmap, evaluate:
    - absolute error: absolute difference between diffusion computed on the unreduced gmap and
                      diffusion computed on the reduced one
    - relative error: same as absolute error but relative
    - time used to reduce: time used to reduce the gmap
    - time used to compute: time used to compute the gmap

    """

    """
    With that image is moreover useful to observe the reasons why the performance are
    better with wave propagation that dijkstra algorithm with certain parameters.
    """

    # For the moment I am using only one image
    logger.info(f"Image name: {image_path}")
    image = cv2.imread(image_path, 0)  # the second parameter with value 0 is needed to read the greyscale image
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


def evaluate_performance_all_dataset(dataset_path: str, image_reduction_factor: int) -> None:
    """
    It executes for each image in dataset path the diffusion distance.
    The computation is performed multiple times with different parameters.
    The error between the unreduced and reduced gmaps are computed.
    The time performance is also computed.

    The results are aggregated for all the images.
    """

    image_paths = os.listdir(dataset_path)

    logger.info("***** RESULTS FOR EACH IMAGE *****\n")

    logger.info(f"Image reduction factor: {image_reduction_factor}\n")

    image_count = 0
    for image_path in image_paths:
        image = cv2.imread(dataset_path + image_path, 0)
        reduced_image = reduce_image_size(image, image_reduction_factor)

        logger.info(f"***** IMAGE {image_count + 1} *****\n")

        logger.info(f"Image name: {image_path}")
        logger.info("{: <20} {: <20}".format("Shape", "Shape reduced"))
        logger.info("{: <20} {: <20}".format(f"{image.shape}", f"{reduced_image.shape}"))

        logger.info("{: <20} {: <20} {: <20} {: <20} {: <20} {: <20}".format("Reduction factor", "Use weights",
                                                                             "Diffusion distance", "Time reduce gmap s",
                                                                             "Time compute dt s", "Time compute distance s"))
        evaluate_performance(image=reduced_image, out_images_path=None, verbose=True,
                             compute_voronoi_diagram=False, reduction_factor=0, use_weights=False)
        evaluate_performance(image=reduced_image, out_images_path=None, verbose=True,
                             compute_voronoi_diagram=False, reduction_factor=0.25, use_weights=False)
        evaluate_performance(image=reduced_image, out_images_path=None, verbose=True,
                             compute_voronoi_diagram=False, reduction_factor=0.25, use_weights=True)
        evaluate_performance(image=reduced_image, out_images_path=None, verbose=True,
                             compute_voronoi_diagram=False, reduction_factor=0.5, use_weights=False)
        evaluate_performance(image=reduced_image, out_images_path=None, verbose=True,
                             compute_voronoi_diagram=False, reduction_factor=0.5, use_weights=True)
        evaluate_performance(image=reduced_image, out_images_path=None, verbose=True,
                             compute_voronoi_diagram=False, reduction_factor=1, use_weights=False)
        evaluate_performance(image=reduced_image, out_images_path=None, verbose=True,
                             compute_voronoi_diagram=False, reduction_factor=1, use_weights=True)

        logger.info("")

        image_count += 1

    logger.info("\n***** AGGREGATE REPORT *****\n")
    logger.info("Number of images")
    logger.info(f"{image_count}")


def main():
    """
    image_path = "../data/time_1/cross/DEHYDRATION_small_leaf_4_time_1_ax1cros_0950_Label_1152x1350_uint8.png"
    evaluate_performance_one_image(image_path)
    """

    evaluate_performance_all_dataset("../data/time_1/cross/", image_reduction_factor=11)


if __name__ == "__main__":
    main()
