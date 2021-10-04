from unittest import TestCase

from distance_transform.pyramidal_dt import *
from distance_transform.dt_utils import *
import numpy as np


class TestPyramidalDt(TestCase):
    def setUp(self) -> None:
        self.binary_image_1 = np.array(
            [[1, 1, 1, 1, 0, 1, 1, 0],
             [1, 1, 1, 0, 1, 1, 1, 1],
             [0, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 0],
             [1, 1, 1, 1, 1, 1, 1, 0],
             [1, 1, 1, 1, 1, 1, 1, 0],
             [1, 1, 1, 1, 0, 1, 1, 0]])
        self.expected_reduced_binary_image_1 = np.array(
            [[1, 0, 0, 0],
             [0, 1, 1, 0],
             [1, 1, 1, 0],
             [1, 1, 0, 0]])
        self.expected_reduced_binary_image_1_dt = np.array(
            [[1, 0, 0, 0],
             [0, 1, 1, 0],
             [1, 2, 1, 0],
             [2, 1, 0, 0]])
        self.expected_interpolated_binary_image_1 = np.array(
            [[2, 1, 0, 1, 0, 1, 0, 1],
             [1, 2, 1, 2, 1, 2, 1, 2],
             [0, 1, 2, 3, 2, 1, 0, 1],
             [1, 2, 3, 4, 3, 2, 1, 2],
             [2, 3, 4, 3, 2, 1, 0, 1],
             [3, 4, 3, 2, 1, 2, 1, 2],
             [4, 3, 2, 1, 0, 1, 0, 1],
             [5, 4, 3, 2, 1, 2, 1, 2]])
        self.expected_binary_image_1_dt = np.array(
            [[2, 3, 2, 1, 0, 1, 1, 0],
             [1, 2, 1, 0, 1, 2, 2, 1],
             [0, 1, 2, 1, 2, 2, 1, 1],
             [1, 2, 3, 2, 2, 1, 0, 0],
             [2, 3, 4, 3, 3, 2, 1, 0],
             [3, 4, 4, 3, 2, 2, 1, 0],
             [4, 4, 3, 2, 1, 2, 1, 0],
             [4, 3, 2, 1, 0, 1, 1, 0]])

        self.binary_image_2 = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 0, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1]])
        self.expected_binary_image_2_dt = np.array(
            [[7, 6, 5, 4, 5, 6, 7, 8],
             [6, 5, 4, 3, 4, 5, 6, 7],
             [5, 4, 3, 2, 3, 4, 5, 6],
             [4, 3, 2, 1, 2, 3, 4, 5],
             [3, 2, 1, 0, 1, 2, 3, 4],
             [4, 3, 2, 1, 2, 3, 4, 5],
             [5, 4, 3, 2, 3, 4, 5, 6],
             [6, 5, 4, 3, 4, 5, 6, 7]])

    def test_reduce_size_binary_image(self):
        actual = reduce_size_binary_image(self.binary_image_1, stride=2)
        self.assertEqual(self.expected_reduced_binary_image_1.tolist(), actual.tolist())

    def test_interpolate_dt_binary_image(self):
        actual = interpolate_dt_binary_image(self.expected_reduced_binary_image_1_dt, stride=2)
        self.assertEqual(self.expected_interpolated_binary_image_1.tolist(), actual.tolist())

    def test_pyramidal_dt_binary_image(self):
        plot_binary_image(self.binary_image_1)
        plot_dt_image(self.expected_binary_image_1_dt, max_value=10)
        actual = pyramidal_dt_binary_image(self.binary_image_1, stride=2)
        self.assertEqual(self.expected_interpolated_binary_image_1.tolist(), actual.tolist())
        plot_dt_image(actual, max_value=10)

    def test_pyramidal_dt_binary_image_single_center(self):
        plot_binary_image(self.binary_image_2)
        plot_dt_image(self.expected_binary_image_2_dt, max_value=10)
        actual = pyramidal_dt_binary_image(self.binary_image_2, stride=2)
        plot_dt_image(actual, max_value=10)

