from cv2 import imread
import numpy as np

from distance_transform.wave_propagation import *
from distance_transform.preprocessing import *
from combinatorial.pixelmap import LabelMap
from combinatorial.zoo_labels import *
from distance_transform.sample_data import *
from distance_transform.dt_utils import *
from distance_transform.dt_applications import *
from visual_test.test_dt_applications import *
import imageio
from combinatorial.utils import build_dt_grey_image_from_gmap
import cv2


def dt_gmap_example():
    print("pippo")

    from combinatorial.pixelmap import PixelMap

    # bounded pixel map example
    bpm = PixelMap.from_shape(2, 2)
    # bpm.plot_faces()
    print(bpm.n_rows)

    for dart in bpm.darts:
        print(type(dart))
        break

    wave_propagation_dt_gmap(bpm, [2, 4, 7])

    # for dart in bpm.darts:
    #     print(type(dart))
    #     dart.identifier
    #     print(dart.attributes["distance"])
    #     break

    bpm.plot_faces_dt()

    """
    from combinatorial.zoo_labels import L2_SPIRAL_WB, str2labels

    print (L2_SPIRAL_WB)

    from combinatorial.pixelmap import LabelMap

    image = str2labels (L2_SPIRAL_WB)
    print(image.shape)

    lm_spiral = LabelMap.from_labels (image)
    print(lm_spiral)

    lm_spiral.remove_edges()
    lm_spiral.remove_vertices()
    print(lm_spiral)

    for dart in lm_spiral.darts:
        print(type(dart))
        print(dart)

    lm_spiral.plot(number_darts=False)
    """


def label_map_example():
    print(CELL_10)

    image = str2labels(CELL_10)
    print(image.shape)

    lm_spiral = LabelMap.from_labels(image)
    lm_spiral.plot(attribute_to_show=False)
    print(lm_spiral)

    lm_spiral.remove_edges()
    lm_spiral.remove_vertices()
    print(lm_spiral)
    lm_spiral.plot(attribute_to_show=True)

    return lm_spiral


def dt_on_label_map_example():
    gmap = label_map_example()
    wave_propagation_dt_gmap(gmap, [9, 191])
    gmap.plot_faces_dt()
    gmap.plot(attribute_to_show=True)


def dt_cell_10():
    image = str2labels(CELL_5)
    # plt.imshow(image)
    # plt.show()

    # Now I need a function to set seeds on an image
    # Then I need a function to convert the image with the seeds
    # to the gmap
    # Otherwise I can choose the seeds later, just considering the
    # vertices

    gmap = LabelMap.from_labels(image)
    gmap.plot()
    # gmap.plot_labels()

    seeds = [6, 5, 40, 47, 148, 185, 157, 192]

    wave_propagation_dt_gmap(gmap, seeds)
    gmap.plot_dt()

    gmap.remove_edges()
    gmap.remove_vertices()
    # gmap.plot(number_darts=True)
    gmap.plot_dt()


def read_leaf_image(path, show: bool = False):
    # Use 0 to read image in grayscale mode
    # I have for sure grayscale images because the image is labeled, so it is not useful to have more that 3 channels
    img = imageio.imread(path)
    # img = cv2.imread(path, 0)
    if show:
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


def build_gmap_from_leaf_image(path):
    img = read_leaf_image(path)
    gmap = LabelMap.from_labels(img)
    print("gmap created successfully")
    # gmap.plot()
    return gmap


def save_labels(path):
    gmap = build_gmap_from_leaf_image(path)
    gmap.plot_labels()


def test_norm_image(path):
    # read gray level image
    image = cv2.imread(path, 0)
    show_image(image)
    # find borders
    image_borders = find_borders(image, 152)
    show_image(image_borders)
    # build gmap
    gmap = LabelMap.from_labels(image_borders)
    # wave propagation
    accumulation_directions = generate_accumulation_directions_cell(2)
    wave_propagation_dt_gmap(gmap, None, accumulation_directions)
    # get image and show results
    dt_image = build_dt_grey_image_from_gmap(gmap)
    plot_dt_image(dt_image, None)


def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_different_values_image(image: np.array) -> None:
    values = set()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            values.add(image[i][j])

    print(values)


def main():
    image = imread(
        "data\image\DEHYDRATION_small_leaf_4_time_1_ax1cros_0950_Label_1152x1350_uint8.png")
    result_img = generalized_wave_propagation_image(
        image=image,
        seed_labels=[labels['stomata']],
        propagation_labels=[labels['air']],
        target_labels=[labels['cell']]
    )
    show_image(result_img)


if __name__ == "__main__":
    main()
