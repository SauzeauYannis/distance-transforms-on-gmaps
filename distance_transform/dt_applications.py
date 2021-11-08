import cv2


def compute_diffusion_distance(image_path: str, save_gmap_image: bool = True) -> None:
    """
    Computes the diffusion distance of the cell represented by the image in image_path.

    Add the parameter save_gmap in order to save the gmap
    Add the parameter reduction_factor in order to compute dt on the reduced gmap
    """

    # Read image
    image = cv2.imred(image_path, 0)  # the second parameter with value 0 is needed to read the greyscale image

    # Compress image

    # Find borders of cells to compute dt and modify the image adding this borders

    # Compute dt from stomata to that cells

    # Average the distance to each point on the border

    # Modify the algorithm to do the same on reduced gmaps
        # I need Dijkstra algorithm

    # Try to improve the segmentation to separate cells and average for each cell
    # not for the borders

    pass
