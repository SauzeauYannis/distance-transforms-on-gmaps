from unittest import TestCase
from distance_transform.preprocessing import *
from data.labels import labels
import cv2
import matplotlib.pyplot as plt


class TestPreprocessing(TestCase):
    def setUp(self) -> None:
        pass

    def test_remove_noise_from_labeled_image(self):
        image = cv2.imread('../data/cross_section_leaf.png', 0)
        values = get_different_values_image(image)
        print(values)

        cleaned_image = remove_noise_from_labeled_image(image, labels)
        values = get_different_values_image(cleaned_image)
        print(values)
        cv2.imwrite('results/cleaned_cross_section_leaf.png', cleaned_image)
        self.assertListEqual(sorted(values), sorted(list(labels.values())))

    def test_clean_borders(self):
        image = cv2.imread('results/cleaned_cross_section_leaf.png', 0)
        cleaned_image = clean_borders(image, 9)
        cv2.imwrite('results/cleaned_borders_cross_section_leaf.png', cleaned_image)
