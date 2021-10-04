from unittest import TestCase
import numpy as np

from distance_transform.performance_evaluation import *


class TestPerformanceEvaluation(TestCase):
    def setUp(self) -> None:
        self.image_1 = np.array(
            [[1, 0, 2, 3],
             [0, 1, 3, 1],
             [1, 1, 1, 0],
             [1, 1, 0, 0]])
        self.image_2 = np.array(
            [[1, 0, 1, 2],
             [0, 1, 3, 1],
             [1, 4, 5, 0],
             [1, 1, 6, 0]])
        self.expected_mae = 0.9375

    def test_mae_image(self):
        actual = mae_image(self.image_1, self.image_2)
        self.assertEqual(self.expected_mae, actual)
