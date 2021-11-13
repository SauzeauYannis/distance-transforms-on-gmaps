from unittest import TestCase
from distance_transform.preprocessing import *
from data.labels import labels
import cv2
import matplotlib.pyplot as plt


class TestPreprocessing(TestCase):
    def setUp(self) -> None:
        pass

    def _matrix_compare(self, expected_matrix: np.array, actual_matrix: np.array) -> True:
        """
        It returns True if the two matrix are equal.
        It returns False otherwise.
        """
        for i in range(expected_matrix.shape[0]):
            for j in range(expected_matrix.shape[1]):
                if expected_matrix[i][j] != actual_matrix[i][j]:
                    return False

        return True

    def test_remove_noise_from_labeled_image(self):
        image = cv2.imread('../data/cross_section_leaf.png', 0)
        values = get_different_values_image(image)
        print(values)

        cleaned_image = remove_noise_from_labeled_image(image, labels)
        values = get_different_values_image(cleaned_image)
        print(values)
        cv2.imwrite('results/cleaned_cross_section_leaf.png', cleaned_image)
        self.assertListEqual(sorted(values), sorted(list(labels.values())))

    def test_generalized_find_borders(self):
        image = cv2.imread("../data/5_5_boundary.png", 0)
        expected = cv2.imread("../data/5_5_boundary_gray_region.png", 0)
        actual = generalized_find_borders(image, 195, 255)

        self.assertEqual(expected.shape, actual.shape)

        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                self.assertEqual(actual[i][j], expected[i][j])

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

    def test_reduce_image_size(self):
        image = cv2.imread("../data/5_5_boundary.png", 0)
        expected = cv2.imread("../data/3_3_boundary_reduced.png", 0)
        actual = reduce_image_size(image, 2)

        self.assertTrue(self._matrix_compare(expected, actual))

    def test_connected_component_labeling_one_pass(self):
        image = cv2.imread("../data/5_5_boundary.png", 0)

        expected = np.zeros(image.shape, dtype=np.uint8)
        expected[1][2] = 1
        expected[1][3] = 1
        expected[1][4] = 1
        expected[2][4] = 1

        expected[2][1] = 2
        expected[3][1] = 2
        expected[4][1] = 2

        expected[2][2] = 3
        expected[2][3] = 3
        expected[3][2] = 3
        expected[3][3] = 3
        expected[3][4] = 3
        expected[4][2] = 3
        expected[4][3] = 3
        expected[4][4] = 3

        actual = connected_component_labeling_one_pass(image)

        print(f"number of labels: {np.max(actual) + 1}")

        self.assertTrue(self._matrix_compare(expected, actual))

