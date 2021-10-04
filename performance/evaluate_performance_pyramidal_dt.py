import typing
from unittest import TestCase

import math

from distance_transform.wave_propagation import wave_propagation_dt_image
from distance_transform.pyramidal_dt import pyramidal_dt_binary_image
from distance_transform.performance_evaluation import mae_image
from distance_transform.dt_utils import *


class EvaluatePerformancePyramidalDt(TestCase):
    def setUp(self) -> None:
        random.seed(42)

    def evaluate_mae_with_multiple_strides(self, image, strides: typing.List[int]) -> None:
        plot_binary_image(image)
        dt_image = wave_propagation_dt_image(image)
        plot_dt_image(dt_image)
        for stride in strides:
            approximate_dt_image = pyramidal_dt_binary_image(image, stride)

            mae = mae_image(dt_image, approximate_dt_image)
            print(f"mae with image_size {image.shape[0]} is {mae} and stride {stride}")

            # Show images
            plot_dt_image(approximate_dt_image)

    def test_mae_one_background_pixel(self):
        image_size = 128
        image = np.ones((image_size, image_size), dtype=int)
        # set central pixel to 0
        image[math.floor(image.shape[0] / 2)][math.floor(image.shape[1] / 2)] = 0

        # evaluate the performance on the same image with different strides (from 2 to 64)
        self.evaluate_mae_with_multiple_strides(image, [2, 4, 8, 16, 64])
        self.assertTrue(True)

    def test_mae_random_images(self):
        image = generate_random_binary_image(128, 0.01)
        # evaluate the performance on the same image with different strides (from 2 to 64)
        self.evaluate_mae_with_multiple_strides(image, [2, 4, 8, 16, 64])
        self.assertTrue(True)
