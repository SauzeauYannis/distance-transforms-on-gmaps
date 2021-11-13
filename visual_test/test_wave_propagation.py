from unittest import TestCase

import numpy as np
from distance_transform.wave_propagation import *
from combinatorial.pixelmap import PixelMap
from distance_transform.dt_utils import *
from combinatorial.pixelmap import LabelMap
from distance_transform.preprocessing import *
import cv2


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

        dt_image = gmap.build_dt_image()
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
