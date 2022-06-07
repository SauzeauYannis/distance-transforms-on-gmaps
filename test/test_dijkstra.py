from combinatorial.pixelmap import LabelMap
from distance_transform.dijkstra import generalized_dijkstra_dt_gmap
from distance_transform.dt_utils import gmap_dt_equal
from distance_transform.wave_propagation import generalized_wave_propagation_gmap, generate_accumulation_directions_cell, generate_accumulation_directions_vertex

from unittest import TestCase

import cv2
import random


class TestDijkstra(TestCase):
    def setUp(self) -> None:
        pass

    def test_generalized_dijkstra_dt_gmap_unweighted_faces(self):
        image = cv2.imread('../data/images/dt_test_image.png', 0)
        actual_gmap = LabelMap.from_labels(image)
        expected_gmap = LabelMap.from_labels(image)

        generalized_dijkstra_dt_gmap(
            actual_gmap, [0], [255], generate_accumulation_directions_cell(2))
        generalized_wave_propagation_gmap(
            expected_gmap, [0], [255], generate_accumulation_directions_cell(2))

        self.assertTrue(gmap_dt_equal(actual_gmap, expected_gmap))

    def test_generalized_dijkstra_dt_gmap_unweighted_vertices(self):
        image = cv2.imread('../data/images/dt_test_image.png', 0)
        actual_gmap = LabelMap.from_labels(image)
        expected_gmap = LabelMap.from_labels(image)

        generalized_dijkstra_dt_gmap(
            actual_gmap, [0], [255], generate_accumulation_directions_vertex(2))
        generalized_wave_propagation_gmap(
            expected_gmap, [0], [255], generate_accumulation_directions_vertex(2))

        self.assertTrue(gmap_dt_equal(actual_gmap, expected_gmap))

    def test_generalized_dijkstra_dt_gmap_weighted_vertices(self):
        random.seed(42)
        image = cv2.imread('../data/images/dt_test_image_2.png', 0)
        actual_gmap = LabelMap.from_labels(image)
        expected_gmap = LabelMap.from_labels(image)

        actual_gmap.remove_edges(0.5)
        actual_gmap.remove_vertices()

        random.seed(42)
        expected_gmap.remove_edges(0.5)
        expected_gmap.remove_vertices()

        for i in range(24):
            expected_gmap.set_dart_distance(i, 0)
        for i in range(40, 56):
            expected_gmap.set_dart_distance(i, 0)

        expected_gmap.set_dart_distance(83, 2)
        expected_gmap.set_dart_distance(93, 2)
        expected_gmap.set_dart_distance(94, 2)
        expected_gmap.set_dart_distance(122, 2)
        expected_gmap.set_dart_distance(128, 2)
        expected_gmap.set_dart_distance(135, 2)

        expected_gmap.set_dart_distance(91, 3)
        expected_gmap.set_dart_distance(92, 3)
        expected_gmap.set_dart_distance(101, 3)
        expected_gmap.set_dart_distance(102, 3)
        expected_gmap.set_dart_distance(129, 3)
        expected_gmap.set_dart_distance(136, 3)

        expected_gmap.set_dart_distance(35, 4)
        expected_gmap.set_dart_distance(36, 4)
        expected_gmap.set_dart_distance(73, 4)
        expected_gmap.set_dart_distance(74, 4)

        expected_gmap.set_dart_distance(163, 4)
        expected_gmap.set_dart_distance(164, 4)
        expected_gmap.set_dart_distance(173, 4)
        expected_gmap.set_dart_distance(174, 4)

        expected_gmap.set_dart_distance(139, 5)
        expected_gmap.set_dart_distance(140, 5)
        expected_gmap.set_dart_distance(149, 5)
        expected_gmap.set_dart_distance(150, 5)
        expected_gmap.set_dart_distance(177, 5)
        expected_gmap.set_dart_distance(178, 5)
        expected_gmap.set_dart_distance(184, 5)
        expected_gmap.set_dart_distance(191, 5)

        expected_gmap.set_dart_distance(75, 5)
        expected_gmap.set_dart_distance(76, 5)
        expected_gmap.set_dart_distance(113, 5)
        expected_gmap.set_dart_distance(114, 5)

        expected_gmap.set_dart_distance(147, 6)
        expected_gmap.set_dart_distance(148, 6)
        expected_gmap.set_dart_distance(157, 6)
        expected_gmap.set_dart_distance(158, 6)
        expected_gmap.set_dart_distance(185, 6)
        expected_gmap.set_dart_distance(186, 6)
        expected_gmap.set_dart_distance(192, 6)
        expected_gmap.set_dart_distance(199, 6)

        expected_gmap.set_dart_distance(179, 6)
        expected_gmap.set_dart_distance(180, 6)
        expected_gmap.set_dart_distance(189, 6)
        expected_gmap.set_dart_distance(190, 6)

        expected_gmap.set_dart_distance(155, 7)
        expected_gmap.set_dart_distance(156, 7)
        expected_gmap.set_dart_distance(193, 7)
        expected_gmap.set_dart_distance(194, 7)

        expected_gmap.set_dart_distance(187, 7)
        expected_gmap.set_dart_distance(188, 7)
        expected_gmap.set_dart_distance(197, 7)
        expected_gmap.set_dart_distance(198, 7)

        actual_gmap.plot()
        actual_gmap.plot(attribute_to_show="weight")

        expected_gmap.plot_dt()

        generalized_dijkstra_dt_gmap(
            actual_gmap, [0], [255], [], generate_accumulation_directions_vertex(2))
        print(generate_accumulation_directions_vertex(2))

        actual_gmap.plot_dt()
        self.assertTrue(gmap_dt_equal(actual_gmap, expected_gmap))

    def test_generalized_dijkstra_dt_gmap_unweighted_faces_propagation_bug(self):
        image = cv2.imread('../data/images/5_5_boundary.png', 0)
        actual_gmap = LabelMap.from_labels(image)
        expected_gmap = LabelMap.from_labels(image)

        # set distances on expected gmap
        for i in range(56):
            expected_gmap.set_dart_distance(i, -1)
        for i in range(56, 80):
            expected_gmap.set_dart_distance(i, 0)
        for i in range(80, 88):
            expected_gmap.set_dart_distance(i, -1)
        for i in range(88, 96):
            expected_gmap.set_dart_distance(i, 0)
        for i in range(96, 112):
            expected_gmap.set_dart_distance(i, 1)
        for i in range(112, 120):
            expected_gmap.set_dart_distance(i, 0)
        for i in range(120, 128):
            expected_gmap.set_dart_distance(i, -1)
        for i in range(128, 136):
            expected_gmap.set_dart_distance(i, 0)
        for i in range(136, 144):
            expected_gmap.set_dart_distance(i, 1)
        for i in range(144, 152):
            expected_gmap.set_dart_distance(i, 2)
        for i in range(152, 160):
            expected_gmap.set_dart_distance(i, 1)
        for i in range(160, 168):
            expected_gmap.set_dart_distance(i, -1)
        for i in range(168, 176):
            expected_gmap.set_dart_distance(i, 0)
        for i in range(176, 184):
            expected_gmap.set_dart_distance(i, 1)
        for i in range(184, 200):
            expected_gmap.set_dart_distance(i, 2)

        generalized_dijkstra_dt_gmap(
            actual_gmap, [0], [195], [], generate_accumulation_directions_cell(2))

        # plot
        expected_gmap.plot()
        expected_gmap.plot_dt(fill_cell='face')
        actual_gmap.plot_dt(fill_cell='face')

        self.assertTrue(gmap_dt_equal(actual_gmap, expected_gmap))
