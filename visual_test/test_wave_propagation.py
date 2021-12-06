from unittest import TestCase

import numpy as np
from distance_transform.wave_propagation import *
from combinatorial.pixelmap import PixelMap
from distance_transform.dt_utils import *
from combinatorial.pixelmap import LabelMap
from distance_transform.preprocessing import *
import cv2
from combinatorial.utils import build_dt_grey_image_from_gmap
from distance_transform.dijkstra import *


class TestWavePropagation(TestCase):
    def setUp(self) -> None:
        self.binary_image_1 = np.array(
            [[1, 1, 1, 1, 0],
             [1, 1, 1, 0, 1],
             [0, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1]])
        self.expected_binary_image_1_dt = np.array(
            [[2, 3, 2, 1, 0],
             [1, 2, 1, 0, 1],
             [0, 1, 2, 1, 2],
             [1, 2, 3, 2, 3],
             [2, 3, 4, 3, 4]])

    def test_dt_after_reductio_big_image(self):
        image = cv2.imread('../data/100_100_portion_leaf.png', 0)
        #image_borders = find_borders(image, 152)
        gmap = LabelMap.from_labels(image)
        gmap.remove_edges()
        gmap.remove_vertices()
        gmap.plot()

        wave_propagation_dt_gmap(gmap, [69593])

        gmap.plot_dt(fill_cell='face')
        gmap.plot_faces_dt()

        dt_image = build_dt_grey_image_from_gmap(gmap)
        plot_dt_image(dt_image, None)

        self.assertTrue(True)

    def test_dt_reduction_05_big_image(self):
        image = cv2.imread('../data/100_100_portion_leaf.png', 0)
        image_borders = find_borders(image, 152)
        compute_dt_reduction(image_borders, 0.5, False)

    def test_dt_reduction_05_300_image(self):
        image = cv2.imread('../data/300_300_portion_leaf.png', 0)
        print("image read successfully")
        image_borders = find_borders(image, 152)
        print("image borders successfully computed")
        compute_dt_reduction(image_borders, 0.5, False)

    def test_build_dt_image_without_interpolation_big_image(self):
        image = cv2.imread('../data/100_100_portion_leaf.png', 0)
        image_borders = find_borders(image, 152)
        compute_dt_reduction(image_borders, 0.5, False, build_image_interpolate=False)

    def test_increase_distance_face_vs_vertex_unweighted(self):
        def _compute_test_dt(gmap, accumulation_directions):
            generalized_wave_propagation_gmap(gmap, [0], [0, 195, 255], accumulation_directions=accumulation_directions)
            gmap.plot_dt(fill_cell=True)
            dt_image = build_dt_grey_image_from_gmap(gmap)
            plot_dt_image(dt_image)

        image = cv2.imread('../data/5_5_boundary.png', 0)
        gmap = LabelMap.from_labels(image)

        # orginal face
        _compute_test_dt(gmap, generate_accumulation_directions_cell(2))

        # original vertex
        _compute_test_dt(gmap, generate_accumulation_directions_vertex(2))

        gmap.remove_edges(0.5)
        gmap.remove_vertices()

        # reduced face
        _compute_test_dt(gmap, generate_accumulation_directions_cell(2))

        # reduced vertex
        _compute_test_dt(gmap, generate_accumulation_directions_vertex(2))

        self.assertTrue(True)

    def test_wave_propagation_labeled_faces(self):
        image = cv2.imread("../data/bug_image_improved.png", 0)
        gmap = LabelMap.from_labels(image)

        accumulation_directions = generate_accumulation_directions_cell(2)
        generalized_wave_propagation_gmap(gmap, [0], [255], [50], accumulation_directions)

        gmap.plot(attribute_to_show=None)
        gmap.plot_dt(fill_cell="face")

        self.assertTrue(True)

    def test_dijkstra_labeled_vertices(self):
        image = cv2.imread("../data/bug_image_improved.png", 0)
        gmap = LabelMap.from_labels(image)

        gmap.plot(attribute_to_show="weight")

        gmap.remove_edge(53)
        gmap.remove_edge(90)
        gmap.remove_edge(18)
        gmap.remove_edge(29)
        gmap.remove_edge(125)
        gmap.remove_edge(122)
        gmap.remove_edge(80)
        gmap.remove_edge(40)
        gmap.remove_edge(176)
        gmap.remove_edge(191)
        gmap.remove_edge(109)
        gmap.remove_edge(118)
        gmap.remove_edge(157)
        gmap.remove_edge(37)

        gmap.remove_edges()

        gmap.remove_vertices()


        generalized_dijkstra_dt_gmap(gmap, [0], [255], generate_accumulation_directions_vertex(2))

        gmap.plot(attribute_to_show=None)
        gmap.plot(attribute_to_show="weight")
        gmap.plot_dt()

        self.assertTrue(True)
