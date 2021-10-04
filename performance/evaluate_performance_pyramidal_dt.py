from unittest import TestCase

import numpy as np
import math

from distance_transform.wave_propagation import wave_propagation_dt_image
from distance_transform.pyramidal_dt import pyramidal_dt_binary_image
from distance_transform.performance_evaluation import mae_image
from distance_transform.dt_utils import *


class EvaluatePerformancePyramidalDt(TestCase):
    def setUp(self) -> None:
        pass

    def test_mae_one_background_pixel(self):
        for i in range(2, 7):
            image_size = 2**i
            stride = 2**(i-1)
            image = np.ones((image_size, image_size), dtype=int)

            # set central pixel to 0
            image[math.floor(image_size / 2)][math.floor(image_size / 2)] = 0

            dt_image = wave_propagation_dt_image(image)
            approximate_dt_image = pyramidal_dt_binary_image(image, stride)

            mae = mae_image(dt_image, approximate_dt_image)
            print(f"mae with image_size {image_size} is {mae} and stride {stride}")

            # Show images
            plot_dt_image(dt_image)
            plot_dt_image(approximate_dt_image)
            """
            """

        self.assertTrue(True)
