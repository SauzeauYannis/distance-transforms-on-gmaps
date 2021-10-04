from unittest import TestCase

import numpy as np
from distance_transform.wave_propagation import *


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
