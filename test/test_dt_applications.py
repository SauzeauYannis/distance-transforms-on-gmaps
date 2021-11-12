from unittest import TestCase
from distance_transform.dt_applications import *
from combinatorial.pixelmap import PixelMap
from distance_transform.preprocessing import *
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
        # Read image
        image = cv2.imread("../data/cleaned_borders_cross_section_leaf.png", 0)  # the second parameter with value 0 is needed to read the greyscale image
        print("image successfully read")
        gmap = compute_dt_for_diffusion_distance(image, "results/dt_diffusion_big_image.png", True)
        diffusion_distance = compute_diffusion_distance(gmap, 50)
        print(f"diffusion_distance: {diffusion_distance}")

    def test_compute_diffusion_distance_reduced_image(self):
        image = cv2.imread("../data/cleaned_borders_cross_section_leaf.png", 0)  # the second parameter with value 0 is needed to read the greyscale image
        print("image successfully read")
        reduced_image = reduce_image_size(image, 5)
        cv2.imwrite("results/reduced_image.png", reduced_image)
        print("image successfully reduced")
        gmap = compute_dt_for_diffusion_distance(reduced_image, "results/dt_diffusion_reduced_image.png", True)
        diffusion_distance = compute_diffusion_distance(gmap, 50)
        print(f"diffusion_distance: {diffusion_distance}")

    def test_compute_diffusion_distance(self):
        expected = 3.0
        actual = compute_diffusion_distance(self.gmap_2_2, 0)
        self.assertEqual(expected, actual)
