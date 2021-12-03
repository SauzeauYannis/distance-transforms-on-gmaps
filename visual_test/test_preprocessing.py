from unittest import TestCase
from distance_transform.preprocessing import *
from data.labels import labels
import cv2
import matplotlib.pyplot as plt


class TestPreprocessing(TestCase):
    def setUp(self) -> None:
        pass

    def test_clean_borders(self):
        image = cv2.imread('results/cleaned_cross_section_leaf.png', 0)
        cleaned_image = clean_borders(image, 9)
        cv2.imwrite('results/cleaned_borders_cross_section_leaf.png', cleaned_image)

    def test_generalized_find_borders(self):
        image = cv2.imread('../data/cleaned_borders_cross_section_leaf.png', 0)
        new_image = generalized_find_borders(image, 0, 50)
        cv2.imwrite('results/borders_image_test.png', new_image)

    def test_connected_component_labeling_one_pass(self):
        """
        """

        image_name = "cleaned_borders_cross_section_leaf.png"
        # image_name = "reduced_image.png"
        image = cv2.imread("../data/" + image_name, 0)
        print("image successfully loaded")
        labeled_image = connected_component_labeling_one_pass(image)
        print("labeled image successfully computed")
        print(f"number of labels: {np.max(labeled_image)}")
        rgb_labeled_image = build_rgb_image_from_labeled_image(labeled_image)
        print("rgb image successfully computed")
        cv2.imwrite("results/labeled_" + image_name, rgb_labeled_image)
        print("rgb image successfully saved")

