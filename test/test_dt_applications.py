from unittest import TestCase

from combinatorial.pixelmap import LabelMap
from distance_transform.dt_applications import compute_diffusion_distance, compute_diffusion_distance_image, compute_dt_for_diffusion_distance, compute_dt_for_diffusion_distance_image
from distance_transform.dt_utils import gmap_dt_equal
from distance_transform.preprocessing import generalized_find_borders, reduce_image_size
from data.labels import labels

import cv2
import numpy as np


class TestPreprocessing(TestCase):
    def setUp(self) -> None:
        pass

    def test_compute_diffusion_distance(self):
        image = cv2.imread('../data/images/2_2_labeled_image.png', 0)
        gmap_2_2 = LabelMap.from_labels(image)

        for i in range(8, 16):
            gmap_2_2.set_dart_distance(i, 2)
        for i in range(24, 32):
            gmap_2_2.set_dart_distance(i, 4)

        expected = 3.0
        actual = compute_diffusion_distance(gmap_2_2, 0)
        self.assertEqual(expected, actual)

    def test_compute_diffusion_distance_image(self):
        image = cv2.imread('../data/images/2_2_labeled_image.png', 0)
        dt_image = np.zeros(image.shape, np.int8)
        dt_image.fill(-1)

        dt_image[0][1] = 2
        dt_image[1][1] = 4

        expected = 3.0
        actual = compute_diffusion_distance_image(image, dt_image, 0)
        self.assertEqual(expected, actual)

    def test_compute_diffusion_distance_negative_weights(self):
        """
        Negative values do not have to be taken into consideration
        in computing diffusion distance
        """

        image = cv2.imread('../data/images/2_2_labeled_image.png', 0)
        gmap_2_2 = LabelMap.from_labels(image)

        for i in range(8, 16):
            gmap_2_2.set_dart_distance(i, 2)
        for i in range(24, 30):
            gmap_2_2.set_dart_distance(i, -1)
        for i in range(30, 32):
            gmap_2_2.set_dart_distance(i, 4)

        expected = 2.4
        actual = compute_diffusion_distance(gmap_2_2, 0)
        self.assertEqual(expected, actual)

    def test_compute_diffusion_distance_negative_weights_image(self):
        """
        Negative values do not have to be taken into consideration
        in computing diffusion distance
        """

        image = cv2.imread('../data/images/2_2_labeled_image.png', 0)
        dt_image = np.zeros(image.shape, np.int8)
        dt_image.fill(-1)

        dt_image[0][1] = 2

        expected = 2.0
        actual = compute_diffusion_distance_image(image, dt_image, 0)
        self.assertEqual(expected, actual)

    def test_compute_dt_for_diffusion_distance_with_without_weights(self):
        """
        If the gmap is not reduced, using or not using the weight should produce the same result.
        """
        image = cv2.imread(
            "../data/images/DEHYDRATION_small_leaf_4_time_1_ax1cros_0950_Color_1152x1350_uint8.png", 0)
        reduced_image = reduce_image_size(image, 11)
        print(f"reduced_image shape: {reduced_image.shape}")
        gmap_without_weights, _, _ = compute_dt_for_diffusion_distance(
            reduced_image, None, True, False, 0, False)
        gmap_with_weights, _, _ = compute_dt_for_diffusion_distance(
            reduced_image, None, True, False, 0, True)

        self.assertTrue(gmap_dt_equal(gmap_without_weights, gmap_with_weights))

        diffusion_distance_without_weights = compute_diffusion_distance(
            gmap_without_weights, labels["cell"])
        diffusion_distance_with_weights = compute_diffusion_distance(
            gmap_with_weights, labels["cell"])

        print(diffusion_distance_without_weights)
        print(diffusion_distance_with_weights)

        self.assertEqual(diffusion_distance_without_weights,
                         diffusion_distance_with_weights)

    def test_compute_dt_for_diffusion_distance_with_without_weights_reduced(self):
        """
        If the gmap is not reduced, using or not using the weight could produce a different result
        """

        image_name = "DEHYDRATION_small_leaf_4_time_1_ax1cros_0950_Label_1152x1350_uint8.png"
        image = cv2.imread("../data/images/" + image_name, 0)
        reduced_image = reduce_image_size(image, 11)
        print(f"reduced_image shape: {reduced_image.shape}")
        gmap_without_weights, _, _ = compute_dt_for_diffusion_distance(
            reduced_image, None, True, False, 0.5, False)
        gmap_with_weights, _, _ = compute_dt_for_diffusion_distance(
            reduced_image, None, True, False, 0.5, True)

        self.assertFalse(gmap_dt_equal(
            gmap_without_weights, gmap_with_weights))

        diffusion_distance_without_weights = compute_diffusion_distance(
            gmap_without_weights, labels["cell"])
        diffusion_distance_with_weights = compute_diffusion_distance(
            gmap_with_weights, labels["cell"])

        self.assertNotEqual(diffusion_distance_without_weights,
                            diffusion_distance_with_weights)

    def test_compute_dt_for_diffusion_distance_without_stomata(self):
        image_name = "DEHYDRATION_small_leaf_4_time_1_ax1cros_0100_Label_1152x1350_uint8.png"
        image = cv2.imread("../data/images/" + image_name, 0)
        reduced_image = reduce_image_size(image, 11)
        print(f"reduced_image shape: {reduced_image.shape}")
        gmap, _, _ = compute_dt_for_diffusion_distance(
            reduced_image, None, True, False, 0.5, False)

        """
        If there are no stomata or there are no cells, the diffusion distance should be equal to -1
        """
        expected = -1
        actual = compute_diffusion_distance(gmap, labels["cell"])

        self.assertEqual(expected, actual)

    def test_compute_dt_for_diffusion_distance_without_stomata_image(self):
        image_name = "DEHYDRATION_small_leaf_4_time_1_ax1cros_0100_Label_1152x1350_uint8.png"
        image = cv2.imread("../data/images/" + image_name, 0)
        reduced_image = reduce_image_size(image, 11)
        print(f"reduced_image shape: {reduced_image.shape}")
        # Find borders
        image_with_borders = generalized_find_borders(
            reduced_image, labels["cell"], labels["cell"])
        dt_image, _ = compute_dt_for_diffusion_distance_image(
            image_with_borders)

        """
        If there are no stomata or there are no cells, the diffusion distance should be equal to -1
        """
        expected = -1
        actual = compute_diffusion_distance_image(
            image_with_borders, dt_image, labels["cell"])

        self.assertEqual(expected, actual)
