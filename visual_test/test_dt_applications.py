from unittest import TestCase
from distance_transform.dt_applications import *
from combinatorial.pixelmap import PixelMap
from distance_transform.preprocessing import *
from data.labels import labels
import cv2


class TestPreprocessing(TestCase):
    def setUp(self) -> None:
        pass

    def test_compute_dt_for_dissufion_distance(self):
        compute_dt_for_diffusion_distance("../data/cleaned_borders_cross_section_leaf.png",
                                          "results/dt_image_cross_sections.png", True)

        self.assertTrue(True)

    def test_compute_diffusion_distance_big_image(self):
        # Read image
        image = cv2.imread("../data/cleaned_borders_cross_section_leaf.png", 0)  # the second parameter with value 0 is needed to read the greyscale image
        print("image successfully read")
        gmap, _, _ = compute_dt_for_diffusion_distance(image, "results/dt_diffusion_big_image.png", True)
        diffusion_distance = compute_diffusion_distance(gmap, 50)
        print(f"diffusion_distance: {diffusion_distance}")

        self.assertTrue(True)

    def test_compute_diffusion_distance_reduced_image(self):
        image = cv2.imread("../data/cleaned_borders_cross_section_leaf.png", 0)  # the second parameter with value 0 is needed to read the greyscale image
        print("image successfully read")
        reduced_image = reduce_image_size(image, 5)
        cv2.imwrite("results/reduced_image.png", reduced_image)
        print("image successfully reduced")
        gmap, _, _ = compute_dt_for_diffusion_distance(reduced_image, "results/dt_diffusion_reduced_image.png",
                                                 True, compute_voronoi_diagram=True)

        dt_voronoi_diagram = gmap.generate_dt_voronoi_diagram([labels["stomata"]])
        print("voronoi diagram successfully generated")

        cv2.imwrite("results/dt_voronoi_diagram.png", dt_voronoi_diagram)

        diffusion_distance = compute_diffusion_distance(gmap, 50)
        print(f"diffusion_distance: {diffusion_distance}")

        self.assertTrue(True)

    def test_generate_dt_voronoi_diagram_image_multiple_stomata(self):
        image_name = "DEHYDRATION_small_leaf_4_time_1_ax1cros_0950_Label_1152x1350_uint8.png"
        image = cv2.imread("../data/time_1/cross/" + image_name, 0)  # the second parameter with value 0 is needed to read the greyscale image
        reduced_image = reduce_image_size(image, 5)
        cv2.imwrite("results/reduced_image_" + image_name, reduced_image)
        gmap, _, _ = compute_dt_for_diffusion_distance(reduced_image, "results/dt_diffusion_reduced_image_" + image_name,
                                                 True, compute_voronoi_diagram=True)
        dt_voronoi_diagram = gmap.generate_dt_voronoi_diagram([labels["stomata"]])
        cv2.imwrite("results/dt_voronoi_diagram_" + image_name, dt_voronoi_diagram)
        diffusion_distance = compute_diffusion_distance(gmap, 50)
        print(f"diffusion_distance: {diffusion_distance}")

        self.assertTrue(True)

    def test_generate_dt_voronoi_diagram_image_multiple_stomata_reduced_gmap(self):
        image_name = "DEHYDRATION_small_leaf_4_time_1_ax1cros_0950_Label_1152x1350_uint8.png"
        image = cv2.imread("../data/time_1/cross/" + image_name, 0)
        reduced_image = reduce_image_size(image, 5)
        gmap, _, _ = compute_dt_for_diffusion_distance(reduced_image, "results/dt_diffusion_reduced_image_" + image_name,
                                                 True, compute_voronoi_diagram=True, reduction_factor=0.5)
        dt_voronoi_diagram = gmap.generate_dt_voronoi_diagram([labels["stomata"]])
        cv2.imwrite("results/dt_voronoi_diagram_reduced_gmap_" + image_name, dt_voronoi_diagram)
        diffusion_distance = compute_diffusion_distance(gmap, 50)
        print(f"diffusion_distance: {diffusion_distance}")

        self.assertTrue(True)

