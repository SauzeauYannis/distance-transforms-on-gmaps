import cv2
from distance_transform.preprocessing import generalized_find_borders
from data.labels import labels
from distance_transform.wave_propagation import generalized_wave_propagation_gmap
from combinatorial.pixelmap import LabelMap


def compute_diffusion_distance(gmap, label: int) -> float:
    """
    It's the average of the distance of each point with label equal to the label passed as parameter

    At the moment I am computing the average on darts
    Probably I should compute the average on faces. It is not so difficult to do that
    """

    sum = 0
    number_of_values = 0

    for dart in gmap.darts:
        if gmap.image_labels[dart] == label:
            number_of_values += 1
            sum += gmap.distances[dart]

    average = sum / number_of_values
    return average


def compute_dt_for_diffusion_distance(image_path: str, dt_image_path: str = None, verbose: bool = False):
    """
    Computes the diffusion distance of the cell represented by the image in image_path.

    Add the parameter save_gmap in order to save the gmap
    Add the parameter reduction_factor in order to compute dt on the reduced gmap
    """

    # Read image
    image = cv2.imread(image_path, 0)  # the second parameter with value 0 is needed to read the greyscale image
    if verbose:
        print("image successfully read")

    # Compress image

    # Find borders of cells to compute dt and modify the image adding this borders
    border_label = 50
    image_with_borders = generalized_find_borders(image, labels["cell"], border_label)
    if verbose:
        print("image with borders successfully computed")

    # build gmap
    gmap = LabelMap.from_labels(image_with_borders)
    if verbose:
        print("gmap successfully builded")

    # Compute dt from stomata to that cells
    propagation_labels = [labels["air"], border_label]
    generalized_wave_propagation_gmap(gmap, [labels["stomata"]], propagation_labels)
    if verbose:
        print("dt successfully computed")

    # save dt image
    dt_image = gmap.build_dt_image()
    if verbose:
        print("dt image successfully computed")

    if dt_image_path:
        cv2.imwrite(dt_image_path, dt_image)

    return gmap

    # Average the distance to each point on the border

    # Modify the algorithm to do the same on reduced gmaps
    # I need Dijkstra algorithm

    # Try to improve the segmentation to separate cells and average for each cell
    # not for the borders
