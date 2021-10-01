from unittest import TestCase

from distance_transform.pyramidal_dt import *
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
        self.expected_reduced_binary_image_stride_2 = np.array(
            [[1, 0, 0, 0],
             [0, 1, 1, 0],
             [1, 1, 1, 0],
             [1, 1, 0, 0]])

    def test_reduce_size_binary_image(self):
        actual = reduce_size_binary_image(self.binary_image_1, stride=2)
        self.assertEqual(self.expected_reduced_binary_image_stride_2.tolist(), actual.tolist())

