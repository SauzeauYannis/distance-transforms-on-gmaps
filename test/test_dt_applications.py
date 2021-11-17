from unittest import TestCase
from distance_transform.dt_applications import *
from combinatorial.pixelmap import PixelMap
from distance_transform.preprocessing import *
import cv2
from combinatorial.utils import build_dt_grey_image


class TestPreprocessing(TestCase):
    def setUp(self) -> None:
        pass

    def test_compute_diffusion_distance(self):
        image = cv2.imread('../data/2_2_labeled_image.png', 0)
        gmap_2_2 = LabelMap.from_labels(image)

        for i in range(8, 16):
            gmap_2_2.set_dart_distance(i, 2)
        for i in range(24, 32):
            gmap_2_2.set_dart_distance(i, 4)

        expected = 3.0
        actual = compute_diffusion_distance(gmap_2_2, 0)
        self.assertEqual(expected, actual)
