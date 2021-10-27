from distance_transform.wave_propagation import wave_propagation_dt_gmap
from combinatorial.pixelmap import LabelMap
from combinatorial.zoo_labels import *
from distance_transform.sample_data import *
import matplotlib.pyplot as plt


def dt_gmap_example():
    print("pippo")

    from combinatorial.pixelmap import PixelMap

    # bounded pixel map example
    bpm = PixelMap.from_shape(2, 2)
    # bpm.plot_faces()

    for dart in bpm.darts:
        print(type(dart))
        break

    wave_propagation_dt_gmap(bpm, [2, 4, 7])

    for dart in bpm.darts_with_attributes:
        print(type(dart))
        dart.identifier
        print(dart.attributes["distance"])
        break

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
    lm_spiral.plot(number_darts=False)
    print(lm_spiral)

    lm_spiral.remove_edges()
    lm_spiral.remove_vertices()
    print(lm_spiral)
    lm_spiral.plot(number_darts=True)

    return lm_spiral


def dt_on_label_map_example():
    gmap = label_map_example()
    wave_propagation_dt_gmap(gmap, [9, 191])
    gmap.plot_faces_dt()
    gmap.plot(number_darts=True)


def dt_cell_10():
    image = str2labels(CELL_5)
    plt.imshow(image)
    plt.show()

    # Now I need a function to set seeds on an image
    # Then I need a function to convert the image with the seeds
    # to the gmap
    # Otherwise I can choose the seeds later, just considering the
    # vertices

    gmap = LabelMap.from_labels(image)
    gmap.plot(number_darts=True)

    seeds = [6, 5, 40, 47, 148, 185, 157, 192]

    wave_propagation_dt_gmap(gmap, seeds)
    gmap.plot_faces_dt()

dt_cell_10()
