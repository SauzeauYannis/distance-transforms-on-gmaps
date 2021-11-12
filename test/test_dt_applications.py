from unittest import TestCase
from distance_transform.dt_applications import *
from combinatorial.pixelmap import PixelMap
import cv2


class TestPreprocessing(TestCase):
    def setUp(self) -> None:
        image = cv2.imread('../data/2_2_labeled_image.png', 0)
        self.gmap_2_2 = LabelMap.from_labels(image)

        for i in range(8, 16):
            self.gmap_2_2.set_dart_distance(i, 2)
        for i in range(24, 32):
            self.gmap_2_2.set_dart_distance(i, 4)

    def test_compute_dt_for_dissufion_distance(self):
        compute_dt_for_diffusion_distance("../data/cleaned_borders_cross_section_leaf.png",
                                          "results/dt_image_cross_sections.png", True)

    def test_compute_diffusion_distance_big_image(self):
        gmap = compute_dt_for_diffusion_distance("../data/cleaned_borders_cross_section_leaf.png", None, True)
        diffusion_distance = compute_diffusion_distance(gmap, 50)
        print(f"diffusion_distance: {diffusion_distance}")

    def test_compute_diffusion_distance(self):
        expected = 3.0
        actual = compute_diffusion_distance(self.gmap_2_2, 0)
        self.assertEqual(expected, actual)
