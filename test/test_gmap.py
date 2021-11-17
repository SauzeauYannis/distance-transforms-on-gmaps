from unittest import TestCase

import numpy as np
from distance_transform.wave_propagation import *
from combinatorial.pixelmap import PixelMap
from distance_transform.dt_utils import *
from combinatorial.pixelmap import LabelMap
from distance_transform.preprocessing import *
from test_utils import *
import cv2
import random


class TestWavePropagation(TestCase):
    def setUp(self) -> None:
        pass

    def test_build_dt_image(self):
        image = cv2.imread('../data/5_5_boundary.png', 0)
        gmap = LabelMap.from_labels(image)
        generalized_wave_propagation_gmap(gmap, [0], [0, 195, 255],
                                          accumulation_directions=generate_accumulation_directions_cell(2))
        dt_image = gmap.build_dt_image()

        self.assertEqual(dt_image.shape, image.shape)
        expected = np.zeros(image.shape, dtype=dt_image.dtype)
        expected[0][0] = 3
        expected[0][1] = 2
        expected[0][2] = 1
        expected[0][3] = 1
        expected[0][4] = 1

        expected[1][0] = 2
        expected[1][1] = 1

        expected[2][0] = 1
        expected[2][2] = 1
        expected[2][3] = 1

        expected[3][0] = 1
        expected[3][2] = 1
        expected[3][3] = 2
        expected[3][4] = 1

        expected[4][0] = 1
        expected[4][2] = 1
        expected[4][3] = 2
        expected[4][4] = 2

        self.assertTrue(matrix_compare(dt_image, expected))

    def test_build_dt_image_no_difference_interpolate(self):
        image = cv2.imread('../data/5_5_boundary.png', 0)
        gmap = LabelMap.from_labels(image)
        generalized_wave_propagation_gmap(gmap, [0], [0, 195, 255],
                                          accumulation_directions=generate_accumulation_directions_cell(2))
        dt_image_interpolate_true = gmap.build_dt_image(True)
        dt_image_interpolate_false = gmap.build_dt_image(False)

        self.assertTrue(matrix_compare(dt_image_interpolate_true, dt_image_interpolate_false))

    def test_build_dt_reduced_image_interpolate_true(self):
        random.seed(42)
        image = cv2.imread('../data/5_5_boundary.png', 0)
        gmap = LabelMap.from_labels(image)
        gmap.remove_edges(0.5)
        gmap.remove_vertices()
        generalized_wave_propagation_gmap(gmap, [0], [0, 195],
                                          accumulation_directions=generate_accumulation_directions_cell(2))
        dt_image = gmap.build_dt_image(interpolate_missing_values=True)

        self.assertEqual(dt_image.shape, image.shape)
        expected = np.zeros(image.shape, dtype=dt_image.dtype)
        expected[0][0] = -2
        expected[0][1] = -2
        expected[0][2] = -2
        expected[0][3] = -2
        expected[0][4] = -2

        expected[1][0] = -2
        expected[1][1] = -2

        expected[2][0] = -2
        expected[2][2] = 1
        expected[2][3] = 1

        expected[3][0] = -2
        expected[3][2] = 1
        expected[3][3] = 2
        expected[3][4] = 1

        expected[4][0] = -2
        expected[4][2] = 1
        expected[4][3] = 1
        expected[4][4] = 1

        print(dt_image)
        print(expected)

        self.assertTrue(matrix_compare(dt_image, expected))

    def test_build_dt_reduced_image_interpolate_false(self):
        random.seed(42)
        image = cv2.imread('../data/5_5_boundary.png', 0)
        gmap = LabelMap.from_labels(image)
        gmap.remove_edges(0.5)
        gmap.remove_vertices()
        generalized_wave_propagation_gmap(gmap, [0], [0, 195],
                                          accumulation_directions=generate_accumulation_directions_cell(2))
        dt_image = gmap.build_dt_image(interpolate_missing_values=False)

        self.assertEqual(dt_image.shape, image.shape)
        expected = np.zeros(image.shape, dtype=dt_image.dtype)
        expected[0][0] = -1
        expected[0][1] = -2
        expected[0][2] = -1
        expected[0][3] = -2
        expected[0][4] = -2

        expected[1][0] = -1
        expected[1][1] = -1
        expected[1][2] = -1

        expected[2][0] = -1
        expected[2][2] = 1
        expected[2][3] = -1
        expected[2][4] = -1

        expected[3][0] = -2
        expected[3][1] = -1
        expected[3][2] = 1
        expected[3][3] = 2
        expected[3][4] = 1

        expected[4][0] = -1
        expected[4][2] = -1
        expected[4][3] = -1
        expected[4][4] = -1

        self.assertTrue(matrix_compare(dt_image, expected))
