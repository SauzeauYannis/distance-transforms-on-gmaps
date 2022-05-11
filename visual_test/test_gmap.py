from unittest import TestCase

import numpy as np
from distance_transform.wave_propagation import *
from combinatorial.pixelmap import PixelMap
from distance_transform.dt_utils import *
from combinatorial.pixelmap import LabelMap
from distance_transform.preprocessing import *
from combinatorial.utils import build_dt_grey_image_from_gmap
from test.test_utils import *
from combinatorial.utils import *
from distance_transform.dt_applications import *
import cv2
import random


class TestGmap(TestCase):
    def setUp(self) -> None:
        pass

    def test_strange_gmap_after_reduction(self):
        """
        The result seems legit and dt is ok, so it seems fine
        """
        random.seed(42)
        image = cv2.imread('../data/dt_test_image.png', 0)
        actual_gmap = LabelMap.from_labels(image)
        actual_gmap.plot(attribute_to_show="weight")

        actual_gmap.remove_edges(0.5)
        actual_gmap.plot()
        actual_gmap.remove_vertices()

        print(actual_gmap.ai(0, 69))
        print(actual_gmap.ai(1, 69))
        print(actual_gmap.ai(2, 69))

        actual_gmap.plot()

        generalized_wave_propagation_gmap(actual_gmap, [0], [255], generate_accumulation_directions_cell(2))
        actual_gmap.plot_dt()

        generalized_wave_propagation_gmap(actual_gmap, [0], [255], generate_accumulation_directions_vertex(2))
        actual_gmap.plot_dt()

        actual_gmap.plot(attribute_to_show="weight")

        self.assertTrue(True)

    def test_contour_plot_from_dt_image(self):
        image_name = "DEHYDRATION_small_leaf_4_time_1_ax1cros_0950_Label_1152x1350_uint8.png"
        image = cv2.imread("../data/time_1/cross/" + image_name, 0)
        reduced_image = reduce_image_size(image, 5)
        gmap, _, _ = compute_dt_for_diffusion_distance(reduced_image, None,
                                                       True, compute_voronoi_diagram=True, use_weights=True)
        dt_image = gmap.build_dt_image()
        dt_grey_image = build_dt_grey_image_from_gmap(gmap)
        plot_dt_image(dt_grey_image)
        contour_plot_from_dt_image(dt_image, 20, 300)
        self.assertTrue(True)

    def test_remove_only_vertices(self):
        image = cv2.imread("../data/2_2_labeled_image.png", 0)
        gmap = LabelMap.from_labels(image)
        gmap.plot()
        gmap.remove_edges()
        gmap.plot()
        """
        for dart in gmap.darts:
            print(dart)
        gmap.plot()
        gmap.remove_edge(5)
        gmap.remove_edge(13)
        gmap.remove_vertex(3)
        gmap.plot()
        # print(list(gmap.all_i_cells(0)))

        self.assertTrue(True)
        """

    def test_white_image(self):
        """
        image = cv2.imread("../data/2_2_white.png", 0)
        gmap = LabelMap.from_labels(image)
        gmap.plot()
        gmap.remove_edges()
        gmap.plot()

        self.assertTrue(True)
        """
        """
        for dart in gmap.darts:
            print(dart)
        gmap.plot()
        gmap.remove_edge(5)
        gmap.remove_edge(13)
        gmap.remove_vertex(3)
        gmap.plot()
        # print(list(gmap.all_i_cells(0)))

        self.assertTrue(True)
        """

        image = cv2.imread("../data/3_3_boundary_reduced.png", 0)
        gmap = LabelMap.from_labels(image)

        accumulation_directions = generate_accumulation_directions_cell(2)
        generalized_wave_propagation_gmap(gmap, [0], [255], [50], accumulation_directions)

        gmap.plot(attribute_to_show="weight")
        gmap.remove_vertex(0)
        gmap.plot(attribute_to_show="weight")
        gmap.plot_dt(fill_cell="face")

        self.assertTrue(True)

    def test_image_generation_after_removal_operations(self):
        image = cv2.imread("../data/time_1/cross/DEHYDRATION_small_leaf_4_time_1_ax1cros_0100_Label_1152x1350_uint8.png", 0)
        reduced_image = reduce_image_size(image, 17)
        gmap = LabelMap.from_labels(reduced_image)
        print("The gmap has been successfully built")
        label_image = gmap.get_label_image()
        plot_color_image(label_image)
        gmap.remove_edges(1.0)
        gmap.remove_vertices()
        print("The edges have been successfully removed")
        label_image = gmap.get_label_image(interpolate_missing_values=True)
        plot_color_image(label_image)
        print("The gmap has been successfully plotted")
        self.assertTrue(True)