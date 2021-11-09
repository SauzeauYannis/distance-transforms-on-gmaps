from unittest import TestCase
from distance_transform.dt_applications import *
import cv2


class TestPreprocessing(TestCase):
    def setUp(self) -> None:
        pass

    def test_dt_applications(self):
        compute_diffusion_distance("../data/cleaned_borders_cross_section_leaf.png",
                                   "results/dt_image_cross_sections.png", True)
