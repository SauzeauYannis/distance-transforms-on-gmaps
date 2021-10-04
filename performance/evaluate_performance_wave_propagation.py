from unittest import TestCase

from distance_transform.dt_utils import *
from distance_transform.wave_propagation import *
from scipy.interpolate import interp1d

import time


class EvaluatePerformanceWavePropagation(TestCase):
    def setUp(self) -> None:
        random.seed(42)

    def test_time_wave_propagation_dt_binary_image(self):
        image_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        elapsed_times = []
        for image_size in image_sizes:
            background_probability = 0.01
            image = generate_random_binary_image(image_size, background_probability)

            start = time.time()
            wave_propagation_dt_image(image)
            end = time.time()

            elapsed_time = end - start
            elapsed_times.append(elapsed_time)

            print(f"image size: {image_size} - number of pixels: {image_size*image_size}"
                  f" background probability: {background_probability} - time s: {elapsed_time}")

        # plot results
        linear = interp1d(image_sizes, elapsed_times)
        cubic = interp1d(image_sizes, elapsed_times, kind='cubic')
        x_new = np.linspace(min(image_sizes), max(image_sizes), num=100, endpoint=True)
        plt.plot(image_sizes, elapsed_times, 'o', x_new, linear(x_new), '-', x_new, cubic(x_new), '--')
        plt.legend(['data', 'linear', 'cubic'], loc='best')
        plt.show()

        self.assertTrue(True)
