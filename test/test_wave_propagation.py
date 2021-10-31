from unittest import TestCase

import numpy as np
from distance_transform.wave_propagation import *
from combinatorial.pixelmap import PixelMap
from distance_transform.dt_utils import *


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

    def test_wave_propagation_dt_binary_image(self):
        actual = wave_propagation_dt_image(self.binary_image_1)
        self.assertEqual(self.expected_binary_image_1_dt.tolist(), actual.tolist())

    def test_wave_propagation_dt_gmap(self):
        seeds = [2, 4, 7]

        actual_gmap = PixelMap.from_shape(2, 2)
        expected_gmap = PixelMap.from_shape(2, 2)

        # set distances on expected gmap
        expected_gmap.set_dart_distance(0, 1)
        expected_gmap.set_dart_distance(1, 1)
        expected_gmap.set_dart_distance(2, 0)
        expected_gmap.set_dart_distance(3, 1)
        expected_gmap.set_dart_distance(4, 0)
        expected_gmap.set_dart_distance(5, 1)
        expected_gmap.set_dart_distance(6, 1)
        expected_gmap.set_dart_distance(7, 0)
        expected_gmap.set_dart_distance(8, 2)
        expected_gmap.set_dart_distance(9, 3)
        expected_gmap.set_dart_distance(10, 4)
        expected_gmap.set_dart_distance(11, 5)
        expected_gmap.set_dart_distance(12, 4)
        expected_gmap.set_dart_distance(13, 3)
        expected_gmap.set_dart_distance(14, 2)
        expected_gmap.set_dart_distance(15, 1)
        expected_gmap.set_dart_distance(16, 2)
        expected_gmap.set_dart_distance(17, 1)
        expected_gmap.set_dart_distance(18, 2)
        expected_gmap.set_dart_distance(19, 3)
        expected_gmap.set_dart_distance(20, 4)
        expected_gmap.set_dart_distance(21, 5)
        expected_gmap.set_dart_distance(22, 4)
        expected_gmap.set_dart_distance(23, 3)
        expected_gmap.set_dart_distance(24, 4)
        expected_gmap.set_dart_distance(25, 5)
        expected_gmap.set_dart_distance(26, 6)
        expected_gmap.set_dart_distance(27, 7)
        expected_gmap.set_dart_distance(28, 6)
        expected_gmap.set_dart_distance(29, 5)
        expected_gmap.set_dart_distance(30, 4)
        expected_gmap.set_dart_distance(31, 3)

        wave_propagation_dt_gmap(actual_gmap, seeds)

        # plot
        expected_gmap.plot_faces()
        expected_gmap.plot_faces_dt()
        actual_gmap.plot_faces_dt()

        self.assertTrue(gmap_dt_equal(actual_gmap, expected_gmap))

    def test_wave_propagation_dt_gmap_vertex(self):
        """
        Test if distance propagates correctly only through vertices
        """
        seeds = [0, 7]

        actual_gmap = PixelMap.from_shape(2, 2)
        expected_gmap = PixelMap.from_shape(2, 2)

        # set distances on expected gmap
        expected_gmap.set_dart_distance(0, 0)
        expected_gmap.set_dart_distance(1, 1)
        expected_gmap.set_dart_distance(2, 1)
        expected_gmap.set_dart_distance(3, 2)
        expected_gmap.set_dart_distance(4, 2)
        expected_gmap.set_dart_distance(5, 1)
        expected_gmap.set_dart_distance(6, 1)
        expected_gmap.set_dart_distance(7, 0)
        expected_gmap.set_dart_distance(8, 1)
        expected_gmap.set_dart_distance(9, 2)
        expected_gmap.set_dart_distance(10, 2)
        expected_gmap.set_dart_distance(11, 3)
        expected_gmap.set_dart_distance(12, 3)
        expected_gmap.set_dart_distance(13, 2)
        expected_gmap.set_dart_distance(14, 2)
        expected_gmap.set_dart_distance(15, 1)
        expected_gmap.set_dart_distance(16, 1)
        expected_gmap.set_dart_distance(17, 2)
        expected_gmap.set_dart_distance(18, 2)
        expected_gmap.set_dart_distance(19, 3)
        expected_gmap.set_dart_distance(20, 3)
        expected_gmap.set_dart_distance(21, 2)
        expected_gmap.set_dart_distance(22, 2)
        expected_gmap.set_dart_distance(23, 1)
        expected_gmap.set_dart_distance(24, 2)
        expected_gmap.set_dart_distance(25, 3)
        expected_gmap.set_dart_distance(26, 3)
        expected_gmap.set_dart_distance(27, 4)
        expected_gmap.set_dart_distance(28, 4)
        expected_gmap.set_dart_distance(29, 3)
        expected_gmap.set_dart_distance(30, 3)
        expected_gmap.set_dart_distance(31, 2)

        accumulation_directions = generate_accumulation_directions_vertex(2)
        wave_propagation_dt_gmap(actual_gmap, seeds, accumulation_directions)

        # plot
        expected_gmap.plot_faces()
        expected_gmap.plot_faces_dt()
        actual_gmap.plot_faces_dt()

        self.assertTrue(gmap_dt_equal(actual_gmap, expected_gmap))

    def test_wave_propagation_dt_gmap_face(self):
        """
        Test if distance propagates correctly only through vertices
        """
        seeds = [0, 1, 2, 3, 4, 5, 6, 7]

        actual_gmap = PixelMap.from_shape(2, 2)
        expected_gmap = PixelMap.from_shape(2, 2)

        # set distances on expected gmap
        expected_gmap.set_dart_distance(0, 0)
        expected_gmap.set_dart_distance(1, 0)
        expected_gmap.set_dart_distance(2, 0)
        expected_gmap.set_dart_distance(3, 0)
        expected_gmap.set_dart_distance(4, 0)
        expected_gmap.set_dart_distance(5, 0)
        expected_gmap.set_dart_distance(6, 0)
        expected_gmap.set_dart_distance(7, 0)
        expected_gmap.set_dart_distance(8, 1)
        expected_gmap.set_dart_distance(9, 1)
        expected_gmap.set_dart_distance(10, 1)
        expected_gmap.set_dart_distance(11, 1)
        expected_gmap.set_dart_distance(12, 1)
        expected_gmap.set_dart_distance(13, 1)
        expected_gmap.set_dart_distance(14, 1)
        expected_gmap.set_dart_distance(15, 1)
        expected_gmap.set_dart_distance(16, 1)
        expected_gmap.set_dart_distance(17, 1)
        expected_gmap.set_dart_distance(18, 1)
        expected_gmap.set_dart_distance(19, 1)
        expected_gmap.set_dart_distance(20, 1)
        expected_gmap.set_dart_distance(21, 1)
        expected_gmap.set_dart_distance(22, 1)
        expected_gmap.set_dart_distance(23, 1)
        expected_gmap.set_dart_distance(24, 2)
        expected_gmap.set_dart_distance(25, 2)
        expected_gmap.set_dart_distance(26, 2)
        expected_gmap.set_dart_distance(27, 2)
        expected_gmap.set_dart_distance(28, 2)
        expected_gmap.set_dart_distance(29, 2)
        expected_gmap.set_dart_distance(30, 2)
        expected_gmap.set_dart_distance(31, 2)

        accumulation_directions = generate_accumulation_directions_cell(2)
        wave_propagation_dt_gmap(actual_gmap, seeds, accumulation_directions)

        # plot
        expected_gmap.plot_faces()
        expected_gmap.plot_faces_dt()
        actual_gmap.plot_faces_dt()

        self.assertTrue(gmap_dt_equal(actual_gmap, expected_gmap))

    def test_wave_propagation_dt_gmap_corner(self):
        seeds = [7]

        actual_gmap = PixelMap.from_shape(2, 2)

        wave_propagation_dt_gmap(actual_gmap, seeds)

        # plot
        actual_gmap.plot_faces_dt()
