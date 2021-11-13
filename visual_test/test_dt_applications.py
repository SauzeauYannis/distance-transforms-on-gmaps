from unittest import TestCase
from distance_transform.dt_applications import *
from combinatorial.pixelmap import PixelMap
from distance_transform.preprocessing import *
import cv2


class TestPreprocessing(TestCase):
    def setUp(self) -> None:
        pass

    def test_compute_dt_for_dissufion_distance(self):
        compute_dt_for_diffusion_distance("../data/cleaned_borders_cross_section_leaf.png",
                                          "results/dt_image_cross_sections.png", True)

        self.assertTrue()

    def test_compute_diffusion_distance_big_image(self):
        # Read image
        image = cv2.imread("../data/cleaned_borders_cross_section_leaf.png", 0)  # the second parameter with value 0 is needed to read the greyscale image
        print("image successfully read")
        gmap = compute_dt_for_diffusion_distance(image, "results/dt_diffusion_big_image.png", True)
        diffusion_distance = compute_diffusion_distance(gmap, 50)
        print(f"diffusion_distance: {diffusion_distance}")

        self.assertTrue()

    def test_compute_diffusion_distance_reduced_image(self):
        image = cv2.imread("../data/cleaned_borders_cross_section_leaf.png", 0)  # the second parameter with value 0 is needed to read the greyscale image
        print("image successfully read")
        reduced_image = reduce_image_size(image, 5)
        cv2.imwrite("results/reduced_image.png", reduced_image)
        print("image successfully reduced")
        gmap = compute_dt_for_diffusion_distance(reduced_image, "results/dt_diffusion_reduced_image.png", True)
        diffusion_distance = compute_diffusion_distance(gmap, 50)
        print(f"diffusion_distance: {diffusion_distance}")

        self.assertTrue()

