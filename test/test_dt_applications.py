from unittest import TestCase
from distance_transform.dt_applications import *
import cv2


class TestPreprocessing(TestCase):
    def setUp(self) -> None:
        pass

    def test_compute_dt_for_dissufion_distance(self):
        compute_dt_for_diffusion_distance("../data/cleaned_borders_cross_section_leaf.png",
                                          "results/dt_image_cross_sections.png", True)

    def test_compute_diffusion_distance(self):
        gmap = compute_dt_for_diffusion_distance("../data/cleaned_borders_cross_section_leaf.png", None, True)
        diffusion_distance = compute_diffusion_distance(gmap, 50)
        print(f"diffusion_distance: {diffusion_distance}")
