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
from combinatorial.utils import build_dt_grey_image_from_gmap
from distance_transform.preprocessing import *
from distance_transform.dt_applications import *
from copy import deepcopy
import os


# get logger
logger = logging.getLogger("evaluate_performance_compute_diffusion_distance_logger")
logging_configuration.set_logging("results")


def evaluate_performance_image(image: np.array, image_reduction_factor: int) -> typing.Dict:
    random.seed(42)

    dt_image, time_to_compute_dt_s = compute_dt_for_diffusion_distance_image(image)

    start = time.time_ns()
    diffusion_distance = compute_diffusion_distance_image(image, dt_image, labels['cell']) * image_reduction_factor
    end = time.time_ns()
    time_to_compute_diffusion_s = (end - start) / (10 ** 9)

    results_dict = {"reduction_factor": None, "n_darts": image.shape[0] * image.shape[1], "use_weights": None, "diffusion_distance": diffusion_distance,
                    "time_reduce_gmap_s": 0, "time_compute_dt_s": time_to_compute_dt_s, "time_to_compute_diffusion_s": time_to_compute_diffusion_s}

    return results_dict


def evaluate_performance(image: np.array, out_images_path: typing.Optional[str], verbose: bool,
                         compute_voronoi_diagram: bool, reduction_factor: float,
                         use_weights: bool, image_reduction_factor: int):
    random.seed(42)

    if out_images_path:
        dt_image_path = out_images_path + "_dt.png"
    else:
        dt_image_path = None
    gmap, time_to_reduce_gmap_s, time_to_compute_dt_s = compute_dt_for_diffusion_distance(image, dt_image_path, verbose,
                                                                                          compute_voronoi_diagram,
                                                                                          reduction_factor,
                                                                                          use_weights)

    start = time.time_ns()
    diffusion_distance = compute_diffusion_distance(gmap, labels['cell']) * image_reduction_factor
    end = time.time_ns()
    time_to_compute_diffusion_s = (end - start) / (10 ** 9)

    if compute_voronoi_diagram and out_images_path:
        voronoi_image_path = out_images_path + "_voronoi.png"
        dt_voronoi_diagram = gmap.generate_dt_voronoi_diagram([labels["stomata"]])
        cv2.imwrite(voronoi_image_path, dt_voronoi_diagram)

    results_dict = {"reduction_factor": reduction_factor, "use_weights": use_weights, "diffusion_distance": diffusion_distance,
                    "time_reduce_gmap_s": time_to_reduce_gmap_s, "time_compute_dt_s": time_to_compute_dt_s,
                    "time_to_compute_diffusion_s": time_to_compute_diffusion_s, "n_darts": gmap.n_darts}

    return gmap, results_dict


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

    RESULT_HEADER_STRING = "{: <10} {: <10} {: <10} {: <10} {: <10} {: <10}" \
                           " {: <10} {: <10} {: <10} {: <10} {: <10}".format("RF", "ND", "UW", "DD", "DD_E", "DD_RE",
                                                                             "TRG_S", "TCDT_S", "TCDT_S_D", "TCDT_S_IF", "TCDD_S")

    AGGREGATE_RESULT_HEADER_STRING = "{: <10} {: <10} {: <10} {: <10} {: <10} {: <10} {: <10} " \
                                     "{: <10} {: <10} {: <10} {: <10} {: <10} {: <10} {: <10}".\
                                     format("RF", "ND", "UW", "DD", "DD_M_E", "DD_SD_E", "DD_MIN_E", "DD_MAX_E",
                                            "DD_RE", "TRG_S", "TCDT_S", "TCDT_S_D", "TCDT_S_IF", "TCDD_S")

    def log_results(base_results: typing.Dict, results: typing.Dict) -> None:
        reduction_factor = results["reduction_factor"]
        n_darts = results["n_darts"]
        use_weights = results["use_weights"]
        diffusion_distance = results["diffusion_distance"]
        diffusion_distance_absolute_error = results["diffusion_distance"] - base_results["diffusion_distance"]
        diffusion_distance_relative_error = diffusion_distance_absolute_error / base_results["diffusion_distance"]
        time_to_reduce_gmap_s = results["time_reduce_gmap_s"]
        time_to_compute_dt_s = results["time_compute_dt_s"]
        time_compute_dt_s_absolute_difference = results["time_compute_dt_s"] - base_results["time_compute_dt_s"]
        time_compute_dt_s_relative_difference = base_results["time_compute_dt_s"] / results["time_compute_dt_s"]
        time_to_compute_diffusion_s = results["time_to_compute_diffusion_s"]

        logger.info("{: <10} {: <10} {: <10} {: <10.2f} {: <+10.2f} {: <+10.2f}"
                    " {: <10.2f} {: <10.4f}"
                    " {: <+10.4f} {: <10.4f} {: <10.4f}".format(f"{reduction_factor}", n_darts, f"{use_weights}", diffusion_distance,
                                                                diffusion_distance_absolute_error, diffusion_distance_relative_error,
                                                                time_to_reduce_gmap_s, time_to_compute_dt_s,
                                                                time_compute_dt_s_absolute_difference, time_compute_dt_s_relative_difference,
                                                                time_to_compute_diffusion_s))

    def update_aggregate_results(aggregate_results_dict: typing.Dict, base_results: typing.Dict, results: typing.Dict) -> None:
        diffusion_distance_error = results["diffusion_distance"] - base_results["diffusion_distance"]
        diffusion_distance_absolute_error = abs(diffusion_distance_error)
        diffusion_distance_relative_error = diffusion_distance_absolute_error / base_results["diffusion_distance"]
        time_compute_dt_s_absolute_difference = results["time_compute_dt_s"] - base_results["time_compute_dt_s"]
        time_compute_dt_s_relative_difference = base_results["time_compute_dt_s"] / results["time_compute_dt_s"]

        aggregate_results_dict["diffusion_distance_errors"].append(diffusion_distance_error)
        aggregate_results_dict["reduction_factor"] = results["reduction_factor"]
        aggregate_results_dict["n_darts"] = results["n_darts"]
        aggregate_results_dict["use_weights"] = results["use_weights"]
        aggregate_results_dict["diffusion_distance"] += results["diffusion_distance"]
        aggregate_results_dict["diffusion_distance_absolute_error"] += diffusion_distance_absolute_error
        aggregate_results_dict["diffusion_distance_relative_error"] += diffusion_distance_relative_error
        aggregate_results_dict["time_reduce_gmap_s"] += results["time_reduce_gmap_s"]
        aggregate_results_dict["time_compute_dt_s"] += results["time_compute_dt_s"]
        aggregate_results_dict["time_compute_dt_s_absolute_difference"] += time_compute_dt_s_absolute_difference
        aggregate_results_dict["time_compute_dt_s_relative_difference"] += time_compute_dt_s_relative_difference
        aggregate_results_dict["time_to_compute_diffusion_s"] += results["time_to_compute_diffusion_s"]

    def compute_average_aggregate_results(aggregate_results_array: typing.List, num_images: int) -> None:
        for aggregate_results_dict in aggregate_results_array:
            aggregate_results_dict["diffusion_distance"] /= num_images
            aggregate_results_dict["diffusion_distance_absolute_error"] /= num_images
            aggregate_results_dict["diffusion_distance_relative_error"] /= num_images
            aggregate_results_dict["time_reduce_gmap_s"] /= num_images
            aggregate_results_dict["time_compute_dt_s"] /= num_images
            aggregate_results_dict["time_compute_dt_s_absolute_difference"] /= num_images
            aggregate_results_dict["time_compute_dt_s_relative_difference"] /= num_images
            aggregate_results_dict["time_to_compute_diffusion_s"] /= num_images


    image_paths = os.listdir(dataset_path)

    logger.info("***** DIFFUSION DISTANCE COMPUTATION EXPERIMENTATION *****\n")
    logger.info(f"Image reduction factor: {image_reduction_factor}\n")
    logger.info("***** LEGEND *****")
    logger.info("RF: Reduction factor\n"
                "ND: Number of darts\n"
                "UW: Use weights\n"
                "DD: Diffusion distance\n"
                "DD_E: Diffusion distance error (current - base)\n"
                "DD_RE: Diffusion distance relative error (current - base) / base\n"
                "TRG_S: Time reduce gmap s\n"
                "TCDT_S: Time compute dt s\n"
                "TCDT_S_D: Time compute dt s difference (current - base)\n"
                "TCDT_S_IF: Time compute dt s improvement factor (base / current)\n"
                "TCDD_S: Time compute distance s\n"
                "\n"
                "DD_M_E: Diffusion distance mean error\n"
                "DD_SD_E: Diffusion distance standard deviation of error\n"
                "DD_MIN_E: Diffusion distance min error (signed, not absolute)\n"
                "DD_MAX_E: Diffusion distance max error (signed, not absolute)\n")

    logger.info("***** RESULTS FOR EACH IMAGE *****\n")

    image_count = 0
    image_count_with_stomata = 0

    # Initialize aggreagte results dict
    # I should use a class for the aggregate results dict
    aggregate_results_dict = {"reduction_factor": -1, "n_darts": 0, "use_weights": False, "diffusion_distance": 0,
                              "diffusion_distance_absolute_error": 0, "diffusion_distance_relative_error": 0,
                              "time_reduce_gmap_s": 0, "time_compute_dt_s": 0,
                              "time_compute_dt_s_absolute_difference": 0, "time_compute_dt_s_relative_difference": 0,
                              "time_to_compute_diffusion_s": 0, "diffusion_distance_errors": []}
    
    # Initialize aggregate results array
    n_results_for_image = 8
    aggregate_results = []
    for i in range(n_results_for_image):
        aggregate_results.append(deepcopy(aggregate_results_dict))

    total_start = time.time()
    for image_path in image_paths:
        image = cv2.imread(dataset_path + image_path, 0)
        reduced_image = reduce_image_size(image, image_reduction_factor)

        logger.info(f"***** IMAGE {image_count + 1} *****\n")

        logger.info(f"Image name: {image_path}")
        logger.info("{: <20} {: <20}".format("Shape", "Shape reduced"))
        logger.info("{: <20} {: <20}".format(f"{image.shape}", f"{reduced_image.shape}"))

        logger.info(RESULT_HEADER_STRING)

        # The first row contains the results computed directed on the image, without using the gmap
        image_results = evaluate_performance_image(image=reduced_image,
                                                   image_reduction_factor=image_reduction_factor)
        log_results(image_results, image_results)
        if image_results["diffusion_distance"] != -1:
            update_aggregate_results(aggregate_results[0], image_results, image_results)

        out_dir = "images/"
        # create image dir
        image_name = os.path.splitext(image_path)[0]
        image_out_dir = out_dir + image_name
        if not os.path.exists(image_out_dir):
            os.mkdir(image_out_dir)
        _, base_results = evaluate_performance(image=reduced_image, out_images_path=image_out_dir + "/_0", verbose=True,
                                               compute_voronoi_diagram=True, reduction_factor=0, use_weights=False,
                                               image_reduction_factor=image_reduction_factor)

        if base_results["diffusion_distance"] != -1:
            image_count_with_stomata += 1

        log_results(base_results, base_results)
        if base_results["diffusion_distance"] != -1:
            update_aggregate_results(aggregate_results[1], base_results, base_results)
        _, results = evaluate_performance(image=reduced_image, out_images_path=image_out_dir + "/_0.25UW", verbose=True,
                                          compute_voronoi_diagram=True, reduction_factor=0.25, use_weights=False,
                                          image_reduction_factor=image_reduction_factor)
        log_results(base_results, results)
        if results["diffusion_distance"] != -1:
            update_aggregate_results(aggregate_results[2], base_results, results)
        _, results = evaluate_performance(image=reduced_image, out_images_path=image_out_dir + "/_0.25W", verbose=True,
                                          compute_voronoi_diagram=True, reduction_factor=0.25, use_weights=True,
                                          image_reduction_factor=image_reduction_factor)
        log_results(base_results, results)
        if results["diffusion_distance"] != -1:
            update_aggregate_results(aggregate_results[3], base_results, results)
        _, results = evaluate_performance(image=reduced_image, out_images_path=image_out_dir + "/_0.5UW", verbose=True,
                                          compute_voronoi_diagram=True, reduction_factor=0.5, use_weights=False,
                                          image_reduction_factor=image_reduction_factor)
        log_results(base_results, results)
        if results["diffusion_distance"] != -1:
            update_aggregate_results(aggregate_results[4], base_results, results)
        _, results = evaluate_performance(image=reduced_image, out_images_path=image_out_dir + "/_0.5W", verbose=True,
                                          compute_voronoi_diagram=True, reduction_factor=0.5, use_weights=True,
                                          image_reduction_factor=image_reduction_factor)
        log_results(base_results, results)
        if results["diffusion_distance"] != -1:
            update_aggregate_results(aggregate_results[5], base_results, results)
        _, results = evaluate_performance(image=reduced_image, out_images_path=image_out_dir + "/_1UW", verbose=True,
                                          compute_voronoi_diagram=True, reduction_factor=1, use_weights=False,
                                          image_reduction_factor=image_reduction_factor)
        log_results(base_results, results)
        if results["diffusion_distance"] != -1:
            update_aggregate_results(aggregate_results[6], base_results, results)
        _, results = evaluate_performance(image=reduced_image, out_images_path=image_out_dir + "/_1W", verbose=True,
                                          compute_voronoi_diagram=True, reduction_factor=1, use_weights=True,
                                          image_reduction_factor=image_reduction_factor)
        log_results(base_results, results)
        if results["diffusion_distance"] != -1:
            update_aggregate_results(aggregate_results[7], base_results, results)

        logger.info("")

        image_count += 1

    total_end = time.time()
    total_execution_time = total_end - total_start
    compute_average_aggregate_results(aggregate_results, image_count_with_stomata)

    logger.info("\n***** AGGREGATE REPORT *****\n")
    logger.info("{: <20} {: <20} {: <20} {: <20}".format("N. images", "N. images stomata",
                                                         "Tot. exec time s", "Avg exec time image s"))
    logger.info("{: <20} {: <20} {: <20.2f} {: <20.2f}".format(f"{image_count}", f"{image_count_with_stomata}",
                                                               total_execution_time, total_execution_time / image_count))
    logger.info("")

    logger.info(AGGREGATE_RESULT_HEADER_STRING)

    for aggregate_results_dict in aggregate_results:
        reduction_factor = aggregate_results_dict["reduction_factor"]
        use_weights = aggregate_results_dict["use_weights"]
        diffusion_distance = aggregate_results_dict["diffusion_distance"]
        diffusion_distance_mean_error = aggregate_results_dict["diffusion_distance_absolute_error"]
        diffusion_distance_error_std = np.std(aggregate_results_dict["diffusion_distance_errors"])
        diffusion_distance_min_err = np.min(aggregate_results_dict["diffusion_distance_errors"])
        diffusion_distance_max_err = np.max(aggregate_results_dict["diffusion_distance_errors"])
        diffusion_distance_relative_error = aggregate_results_dict["diffusion_distance_relative_error"]
        time_compute_dt_s_absolute_difference = aggregate_results_dict["time_compute_dt_s_absolute_difference"]
        time_compute_dt_s_relative_difference = aggregate_results_dict["time_compute_dt_s_relative_difference"]
        time_to_reduce_gmap_s = aggregate_results_dict["time_reduce_gmap_s"]
        time_to_compute_dt_s = aggregate_results_dict["time_compute_dt_s"]
        time_to_compute_diffusion_s = aggregate_results_dict["time_to_compute_diffusion_s"]
        n_darts = aggregate_results_dict["n_darts"]
        logger.info("{: <10} {: <10} {: <10} {: <10.2f} {: <10.2f} {: <10.2f} {: <+10.2f}"
                    " {: <+10.2f} {: <10.2f} {: <10.2f} {: <10.4f}"
                    " {: <+10.4f} {: <10.4f} {: <10.4f}".format(f"{reduction_factor}", n_darts, f"{use_weights}", diffusion_distance,
                                                                diffusion_distance_mean_error, diffusion_distance_error_std,
                                                                diffusion_distance_min_err, diffusion_distance_max_err, diffusion_distance_relative_error,
                                                                time_to_reduce_gmap_s, time_to_compute_dt_s,
                                                                time_compute_dt_s_absolute_difference, time_compute_dt_s_relative_difference,
                                                                time_to_compute_diffusion_s))


def evaluate_performance_all_dataset_images(dataset_path: str, image_reduction_factor: int) -> None:
    """
    It executes for each image in dataset path the diffusion distance.
    The computation is performed multiple times with different parameters.
    The error between the unreduced and reduced gmaps are computed.
    The time performance is also computed.

    The results are aggregated for all the images.
    """

    RESULT_HEADER_STRING = "{: <10} {: <10} {: <10} {: <10} {: <10}" \
                           " {: <10} {: <10} {: <10} {: <10} {: <10}".format("RF", "UW", "DD", "DD_E", "DD_RE",
                                                                             "TRG_S", "TCDT_S", "TCDT_S_D", "TCDT_S_IF", "TCDD_S")

    def log_results(base_results: typing.Dict, results: typing.Dict) -> None:
        reduction_factor = results["reduction_factor"]
        use_weights = results["use_weights"]
        diffusion_distance = results["diffusion_distance"]
        diffusion_distance_absolute_error = results["diffusion_distance"] - base_results["diffusion_distance"]
        diffusion_distance_relative_error = diffusion_distance_absolute_error / base_results["diffusion_distance"]
        time_to_reduce_gmap_s = results["time_reduce_gmap_s"]
        time_to_compute_dt_s = results["time_compute_dt_s"]
        time_compute_dt_s_absolute_difference = results["time_compute_dt_s"] - base_results["time_compute_dt_s"]
        time_compute_dt_s_relative_difference = base_results["time_compute_dt_s"] / results["time_compute_dt_s"]
        time_to_compute_diffusion_s = results["time_to_compute_diffusion_s"]

        logger.info("{: <10} {: <10} {: <10.2f} {: <+10.2f} {: <+10.2f}"
                    " {: <10.2f} {: <10.4f}"
                    " {: <+10.4f} {: <10.4f} {: <10.4f}".format(f"{reduction_factor}", f"{use_weights}", diffusion_distance,
                                                                diffusion_distance_absolute_error, diffusion_distance_relative_error,
                                                                time_to_reduce_gmap_s, time_to_compute_dt_s,
                                                                time_compute_dt_s_absolute_difference, time_compute_dt_s_relative_difference,
                                                                time_to_compute_diffusion_s))

    def update_aggregate_results(aggregate_results_dict: typing.Dict, base_results: typing.Dict, results: typing.Dict) -> None:
        diffusion_distance_absolute_error = abs(results["diffusion_distance"] - base_results["diffusion_distance"])
        diffusion_distance_relative_error = diffusion_distance_absolute_error / base_results["diffusion_distance"]
        time_compute_dt_s_absolute_difference = results["time_compute_dt_s"] - base_results["time_compute_dt_s"]
        time_compute_dt_s_relative_difference = base_results["time_compute_dt_s"] / results["time_compute_dt_s"]

        aggregate_results_dict["reduction_factor"] = results["reduction_factor"]
        aggregate_results_dict["use_weights"] = results["use_weights"]
        aggregate_results_dict["diffusion_distance"] += results["diffusion_distance"]
        aggregate_results_dict["diffusion_distance_absolute_error"] += diffusion_distance_absolute_error
        aggregate_results_dict["diffusion_distance_relative_error"] += diffusion_distance_relative_error
        aggregate_results_dict["time_reduce_gmap_s"] += results["time_reduce_gmap_s"]
        aggregate_results_dict["time_compute_dt_s"] += results["time_compute_dt_s"]
        aggregate_results_dict["time_compute_dt_s_absolute_difference"] += time_compute_dt_s_absolute_difference
        aggregate_results_dict["time_compute_dt_s_relative_difference"] += time_compute_dt_s_relative_difference
        aggregate_results_dict["time_to_compute_diffusion_s"] += results["time_to_compute_diffusion_s"]

    def compute_average_aggregate_results(aggregate_results_array: typing.List, num_images: int) -> None:
        for aggregate_results_dict in aggregate_results_array:
            aggregate_results_dict["diffusion_distance"] /= num_images
            aggregate_results_dict["diffusion_distance_absolute_error"] /= num_images
            aggregate_results_dict["diffusion_distance_relative_error"] /= num_images
            aggregate_results_dict["time_reduce_gmap_s"] /= num_images
            aggregate_results_dict["time_compute_dt_s"] /= num_images
            aggregate_results_dict["time_compute_dt_s_absolute_difference"] /= num_images
            aggregate_results_dict["time_compute_dt_s_relative_difference"] /= num_images
            aggregate_results_dict["time_to_compute_diffusion_s"] /= num_images


    image_paths = os.listdir(dataset_path)

    logger.info("***** DIFFUSION DISTANCE COMPUTATION EXPERIMENTATION *****\n")
    logger.info(f"Image reduction factor: {image_reduction_factor}\n")
    logger.info("***** LEGEND *****")
    logger.info("RF: Reduction factor\n"
                "UW: Use weights\n"
                "DD: Diffusion distance\n"
                "DD_E: Diffusion distance error (current - base)\n"
                "DD_RE: Diffusion distance relative error (current - base) / base\n"
                "TRG_S: Time reduce gmap s\n"
                "TCDT_S: Time compute dt s\n"
                "TCDT_S_D: Time compute dt s difference (current - base)\n"
                "TCDT_S_IF: Time compute dt s improvement factor (base / current)\n"
                "TCDD_S: Time compute distance s\n")

    logger.info("***** RESULTS FOR EACH IMAGE *****\n")

    image_count = 0
    image_count_with_stomata = 0

    # Initialize aggreagte results dict
    # I should use a class for the aggregate results dict
    aggregate_results_dict = {"reduction_factor": -1, "use_weights": False, "diffusion_distance": 0,
                              "diffusion_distance_absolute_error": 0, "diffusion_distance_relative_error": 0,
                              "time_reduce_gmap_s": 0, "time_compute_dt_s": 0,
                              "time_compute_dt_s_absolute_difference": 0, "time_compute_dt_s_relative_difference": 0,
                              "time_to_compute_diffusion_s": 0}

    # Initialize aggregate results array
    n_results_for_image = 1
    aggregate_results = []
    for i in range(n_results_for_image):
        aggregate_results.append(deepcopy(aggregate_results_dict))

    total_start = time.time()
    for image_path in image_paths:
        image = cv2.imread(dataset_path + image_path, 0)
        reduced_image = reduce_image_size(image, image_reduction_factor)

        logger.info(f"***** IMAGE {image_count + 1} *****\n")

        logger.info(f"Image name: {image_path}")
        logger.info("{: <20} {: <20}".format("Shape", "Shape reduced"))
        logger.info("{: <20} {: <20}".format(f"{image.shape}", f"{reduced_image.shape}"))

        logger.info(RESULT_HEADER_STRING)

        # The first row contains the results computed directed on the image, without using the gmap
        image_results = evaluate_performance_image(image=reduced_image, image_reduction_factor=image_reduction_factor)
        log_results(image_results, image_results)
        if image_results["diffusion_distance"] != -1:
            update_aggregate_results(aggregate_results[0], image_results, image_results)
            image_count_with_stomata += 1

        logger.info("")

        image_count += 1

    total_end = time.time()
    total_execution_time = total_end - total_start
    compute_average_aggregate_results(aggregate_results, image_count_with_stomata)

    logger.info("\n***** AGGREGATE REPORT *****\n")
    logger.info("{: <20} {: <20} {: <20} {: <20}".format("N. images", "N. images stomata",
                                                         "Tot. exec time s", "Avg exec time image s"))
    logger.info("{: <20} {: <20} {: <20.2f} {: <20.2f}".format(f"{image_count}", f"{image_count_with_stomata}",
                                                               total_execution_time, total_execution_time / image_count))
    logger.info("")

    logger.info(RESULT_HEADER_STRING)

    for aggregate_results_dict in aggregate_results:
        reduction_factor = aggregate_results_dict["reduction_factor"]
        use_weights = aggregate_results_dict["use_weights"]
        diffusion_distance = aggregate_results_dict["diffusion_distance"]
        diffusion_distance_absolute_error = aggregate_results_dict["diffusion_distance_absolute_error"]
        diffusion_distance_relative_error = aggregate_results_dict["diffusion_distance_relative_error"]
        time_compute_dt_s_absolute_difference = aggregate_results_dict["time_compute_dt_s_absolute_difference"]
        time_compute_dt_s_relative_difference = aggregate_results_dict["time_compute_dt_s_relative_difference"]
        time_to_reduce_gmap_s = aggregate_results_dict["time_reduce_gmap_s"]
        time_to_compute_dt_s = aggregate_results_dict["time_compute_dt_s"]
        time_to_compute_diffusion_s = aggregate_results_dict["time_to_compute_diffusion_s"]
        logger.info("{: <10} {: <10} {: <10.2f} {: <10.2f} {: <10.2f}"
                    " {: <10.2f} {: <10.2f}"
                    " {: <+10.2f} {: <10.2f} {: <10.2f}".format(f"{reduction_factor}", f"{use_weights}", diffusion_distance,
                                                                diffusion_distance_absolute_error, diffusion_distance_relative_error,
                                                                time_to_reduce_gmap_s, time_to_compute_dt_s,
                                                                time_compute_dt_s_absolute_difference, time_compute_dt_s_relative_difference,
                                                                time_to_compute_diffusion_s))


def evaluate_precision_images_different_resolutions(dir_path: str):
    for i in range(39, 40, 2):
        evaluate_performance_all_dataset_images(dataset_path=dir_path, image_reduction_factor=i)


def main():
    """
    image_path = "../data/time_1/cross/DEHYDRATION_small_leaf_4_time_1_ax1cros_0950_Label_1152x1350_uint8.png"
    evaluate_performance_one_image(image_path)
    """

    # evaluate_precision_images_different_resolutions("../data/labels_with_stomata/time_1/cross/")
    evaluate_performance_all_dataset("../data/labels_with_stomata/long/", image_reduction_factor=11)
    # evaluate_performance_all_dataset("../data/diffusion_distance_images/", image_reduction_factor=11)


if __name__ == "__main__":
    main()
