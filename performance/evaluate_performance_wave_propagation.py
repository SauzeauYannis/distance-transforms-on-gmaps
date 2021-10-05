import typing
from unittest import TestCase

from distance_transform.dt_utils import *
from distance_transform.wave_propagation import *
from scipy.interpolate import interp1d
from combinatorial.pixelmap import PixelMap

import time


class EvaluatePerformanceWavePropagation(TestCase):
    def setUp(self) -> None:
        random.seed(42)

    def plot_size_time(self, sizes: typing.List[int], times: typing.List[float]) -> None:
        linear = interp1d(sizes, times)
        cubic = interp1d(sizes, times, kind='cubic')
        x_new = np.linspace(min(sizes), max(sizes), num=100, endpoint=True)
        plt.plot(sizes, times, 'o', x_new, linear(x_new), '-', x_new, cubic(x_new), '--')
        plt.legend(['data', 'linear', 'cubic'], loc='best')
        plt.show()

    def test_time_wave_propagation_dt_binary_image(self):
        image_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        elapsed_times = []
        background_probability = 0.01

        for image_size in image_sizes:
            image = generate_random_binary_image(image_size, background_probability)

            start = time.time()
            wave_propagation_dt_image(image)
            end = time.time()

            elapsed_time = end - start
            elapsed_times.append(elapsed_time)

            print(f"image size: {image_size} - number of pixels: {image_size*image_size}"
                  f" background probability: {background_probability} - time s: {elapsed_time}")

        # plot results
        self.plot_size_time(image_sizes, elapsed_times)
        self.assertTrue(True)

    def test_time_wave_propagation_dt_gmap(self):
        image_sizes = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        elapsed_times = []
        background_probability = 0.01

        for image_size in image_sizes:
            gmap = PixelMap.from_shape(image_size, image_size)
            # select random seeds
            seeds = []
            for dart in gmap.darts:
                if random.random() < background_probability:
                    seeds.append(dart)

            start = time.time()
            wave_propagation_dt_gmap(gmap, seeds)
            end = time.time()

            elapsed_time = end - start
            elapsed_times.append(elapsed_time)

            print(f"image size: {image_size} - number of darts: {image_size*image_size*8}"
                  f" background probability: {background_probability} - time s: {elapsed_time}")

        # plot results
        self.plot_size_time(image_sizes, elapsed_times)
        self.assertTrue(True)
