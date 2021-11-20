from unittest import TestCase

import numpy as np
from distance_transform.wave_propagation import *
from combinatorial.pixelmap import PixelMap
from distance_transform.dt_utils import *
from combinatorial.pixelmap import LabelMap
from distance_transform.preprocessing import *
from combinatorial.utils import build_dt_grey_image
from test.test_utils import *
from combinatorial.utils import *
import cv2
import random


class TestGmap(TestCase):
    def setUp(self) -> None:
        pass

    def test_strange_gmap_after_reduction(self):
        """
        The result seems legit and dt is ok, so it seems fine
        """
        random.seed(42)
        image = cv2.imread('../data/dt_test_image.png', 0)
        actual_gmap = LabelMap.from_labels(image)

        actual_gmap.remove_edges(0.5)
        actual_gmap.plot()
        actual_gmap.remove_vertices()

        print(actual_gmap.ai(0, 69))
        print(actual_gmap.ai(1, 69))
        print(actual_gmap.ai(2, 69))

        actual_gmap.plot()

        generalized_wave_propagation_gmap(actual_gmap, [0], [255], generate_accumulation_directions_cell(2))
        actual_gmap.plot_dt()

        generalized_wave_propagation_gmap(actual_gmap, [0], [255], generate_accumulation_directions_vertex(2))
        actual_gmap.plot_dt()

        self.assertTrue(True)
