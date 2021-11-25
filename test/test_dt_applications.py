from unittest import TestCase
from distance_transform.dt_applications import *
from combinatorial.pixelmap import PixelMap
from distance_transform.preprocessing import *
import cv2
from combinatorial.utils import build_dt_grey_image


class TestPreprocessing(TestCase):
    def setUp(self) -> None:
        pass

    def test_compute_diffusion_distance(self):
        image = cv2.imread('../data/2_2_labeled_image.png', 0)
        gmap_2_2 = LabelMap.from_labels(image)

        for i in range(8, 16):
            gmap_2_2.set_dart_distance(i, 2)
        for i in range(24, 32):
            gmap_2_2.set_dart_distance(i, 4)

        expected = 3.0
        actual = compute_diffusion_distance(gmap_2_2, 0)
        self.assertEqual(expected, actual)

    def test_compute_diffusion_distance_negative_weights(self):
        """
        Negative values do not have to be taken into consideration
        in computing diffusion distance
        """

        image = cv2.imread('../data/2_2_labeled_image.png', 0)
        gmap_2_2 = LabelMap.from_labels(image)

        for i in range(8, 16):
            gmap_2_2.set_dart_distance(i, 2)
        for i in range(24, 30):
            gmap_2_2.set_dart_distance(i, -1)
        for i in range(30, 32):
            gmap_2_2.set_dart_distance(i, 4)

        gmap_2_2.plot()
        gmap_2_2.plot_dt()

        expected = 2.4
        actual = compute_diffusion_distance(gmap_2_2, 0)
        self.assertEqual(expected, actual)

    def test_compute_dt_for_diffusion_distance_with_without_weights(self):
        """
        If the gmap is not reduced, using or not using the weight should produce the same result.
        """
        image_name = "DEHYDRATION_small_leaf_4_time_1_ax1cros_0950_Label_1152x1350_uint8.png"
        image = cv2.imread("../data/time_1/cross/" + image_name, 0)
        reduced_image = reduce_image_size(image, 11)
        print(f"reduced_image shape: {reduced_image.shape}")
        gmap_without_weights, _, _ = compute_dt_for_diffusion_distance(reduced_image, None, True, False, 0, False)
        gmap_with_weights, _, _ = compute_dt_for_diffusion_distance(reduced_image, None, True, False, 0, True)

        self.assertTrue(gmap_dt_equal(gmap_without_weights, gmap_with_weights))

        diffusion_distance_without_weights = compute_diffusion_distance(gmap_without_weights, 50)
        diffusion_distance_with_weights = compute_diffusion_distance(gmap_with_weights, 50)

        self.assertEqual(diffusion_distance_without_weights, diffusion_distance_with_weights)

    def test_compute_dt_for_diffusion_distance_with_without_weights_reduced(self):
        """
        If the gmap is not reduced, using or not using the weight could produce a different result
        """

        image_name = "DEHYDRATION_small_leaf_4_time_1_ax1cros_0950_Label_1152x1350_uint8.png"
        image = cv2.imread("../data/time_1/cross/" + image_name, 0)
        reduced_image = reduce_image_size(image, 11)
        print(f"reduced_image shape: {reduced_image.shape}")
        gmap_without_weights, _, _ = compute_dt_for_diffusion_distance(reduced_image, None, True, False, 0.5, False)
        gmap_with_weights, _, _ = compute_dt_for_diffusion_distance(reduced_image, None, True, False, 0.5, True)

        self.assertFalse(gmap_dt_equal(gmap_without_weights, gmap_with_weights))

        diffusion_distance_without_weights = compute_diffusion_distance(gmap_without_weights, 50)
        diffusion_distance_with_weights = compute_diffusion_distance(gmap_with_weights, 50)

        self.assertNotEqual(diffusion_distance_without_weights, diffusion_distance_with_weights)
