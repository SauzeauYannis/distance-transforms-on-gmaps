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

    def test_generalized_find_borders(self):
        image = cv2.imread("../data/5_5_boundary.png", 0)
        expected = cv2.imread("../data/5_5_boundary_gray_region.png", 0)
        actual = generalized_find_borders(image, 195, 255)

        self.assertEqual(expected.shape, actual.shape)

        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                self.assertEqual(actual[i][j], expected[i][j])

    def test_generalized_find_borders_big_image(self):
        image = cv2.imread('../data/cleaned_borders_cross_section_leaf.png', 0)
        new_image = generalized_find_borders(image, 0, 50)
        cv2.imwrite('results/borders_image_test.png', new_image)

    def test_is_equal_to_neighbours_true(self):
        image = cv2.imread("../data/5_5_boundary.png", 0)
        expected = True
        actual = is_equal_to_neighbours(image, (3, 3))
        self.assertEqual(expected, actual)

    def test_is_equal_to_neighbours_border(self):
        image = cv2.imread("../data/5_5_boundary.png", 0)
        expected = True
        actual = is_equal_to_neighbours(image, (0, 0))
        self.assertEqual(expected, actual)

    def test_is_equal_to_neighbours_false(self):
        image = cv2.imread("../data/5_5_boundary.png", 0)
        expected = False
        actual = is_equal_to_neighbours(image, (0, 2))
        self.assertEqual(expected, actual)


