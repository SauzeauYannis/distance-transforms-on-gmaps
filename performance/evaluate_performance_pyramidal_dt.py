import typing
from unittest import TestCase

import math

from distance_transform.wave_propagation import wave_propagation_dt_image
from distance_transform.pyramidal_dt import *
from distance_transform.performance_evaluation import mae_image
from distance_transform.dt_utils import *


class EvaluatePerformancePyramidalDt(TestCase):
    def setUp(self) -> None:
        random.seed(42)

        self.image_size = 128
        self.strides = [2, 4, 8, 16, 64]

    def evaluate_mae_with_multiple_strides(self, image, strides: typing.List[int], algorithm: typing.Callable) -> None:
        plot_binary_image(image)
        dt_image = wave_propagation_dt_image(image)
        plot_dt_image(dt_image)
        for stride in strides:
            approximate_dt_image = algorithm(image, stride)

            mae = mae_image(dt_image, approximate_dt_image)
            print(f"mae: {mae} - stride: {stride} - image_compression_percentage: {(1 - 1/stride) * 100}%")

            # Show images
            plot_dt_image(approximate_dt_image)

    def test_mae_one_background_pixel(self):
        image = np.ones((self.image_size, self.image_size), dtype=int)
        # set central pixel to 0
        image[math.floor(image.shape[0] / 2)][math.floor(image.shape[1] / 2)] = 0

        # evaluate the performance on the same image with different strides (from 2 to 64)
        self.evaluate_mae_with_multiple_strides(image, [2, 4, 8, 16, 64])
        self.assertTrue(True)

    def test_mae_random_images_pyramidal(self):
        background_prob = 0.01
        print(f"background probability: {background_prob} - image size: {self.image_size}")
        image = generate_random_binary_image(self.image_size, background_prob)
        self.evaluate_mae_with_multiple_strides(image, self.strides, pyramidal_dt_binary_image)

        background_prob = 0.001
        print(f"background probability: {background_prob} - image size: {self.image_size}")
        image = generate_random_binary_image(self.image_size, background_prob)
        self.evaluate_mae_with_multiple_strides(image, self.strides, pyramidal_dt_binary_image)
        self.assertTrue(True)

    def test_mae_random_images_improved_pyramidal(self):
        background_prob = 0.01
        print(f"background probability: {background_prob} - image size: {self.image_size}")
        image = generate_random_binary_image(self.image_size, background_prob)
        self.evaluate_mae_with_multiple_strides(image, self.strides, improved_pyramidal_dt_binary_image)

        background_prob = 0.001
        print(f"background probability: {background_prob} - image size: {self.image_size}")
        image = generate_random_binary_image(self.image_size, background_prob)
        self.evaluate_mae_with_multiple_strides(image, self.strides, improved_pyramidal_dt_binary_image)
        self.assertTrue(True)
