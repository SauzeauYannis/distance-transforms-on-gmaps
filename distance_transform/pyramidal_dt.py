import typing

import numpy as np
import math

from distance_transform.wave_propagation import wave_propagation_dt_image


def pyramidal_dt_binary_image(image: np.array, stride: int) -> np.array:
    """Parallel approximate algorithm that should (hopefully) takes O(log(n)) where n is the number of darts

    Time complexity: O(log(n)) ?
    Space complexity: O(n) ?

    Algorithm:
    1) Reduce image size through a pyramid
    2) Compute dt on a reduced image
    3) Interpolate the obtained values for each level

    Args:
        image: the image to compute the distance transform
        stride: the stride of the pyramid

    Returns:
        np.array: the distance transform image
    """

    # reduce image
    reduced_image = reduce_size_binary_image(image, stride)

    # compute dt
    dt_image = wave_propagation_dt_image(reduced_image)

    # interpolate
    dt_image = interpolate_dt_binary_image(dt_image, stride)

    return dt_image


def improved_pyramidal_dt_binary_image(image: np.array, stride: int) -> np.array:
    """Improved version of the pyramidal_dt_binary_image algorithm.

    Args:
        image: the image to compute the distance transform
        stride: the stride of the pyramid

    Returns:
        np.array: the distance transform image
    """

    # reduce image
    reduced_image = reduce_size_binary_image(image, stride)

    # compute dt
    dt_image = wave_propagation_dt_image(reduced_image)

    # interpolate
    dt_image = improved_interpolate_dt_binary_image(image, dt_image)

    return dt_image


def reduce_size_binary_image(image: np.array, stride: int) -> np.array:
    """Reduce the size of a binary image with a square kernel of size stride

    How to compress the image?
    A binary image has only zeroes and ones.

    The idea is to use a kernel to compress all the values under the kernel to a unique value in the reduced image.
    How to chose the value?
    1) If all the values under the kernel are the same (all zeroes or all ones) the new value in the reduced image
       is the same value in the original one.
    2) If the values are not the same two different approaches can be pursued:
        a) take the most representative value
        b) take always 0 (seed value). In fact we won't lose the seeds, so it could be a good idea to preserve them
        In either cases we are losing information, so we should be careful.
        Maybe different approaches can be tried.

    How to manage boundaries if the kernel size is not perfect?
    I can add ones to the boundaries if the kernel size does not fit perfectly.

    PRECONDITIONS:
    At the moment, for simplicity, only 2d images are considered.

    Args:
        image: the image to reduce
        stride: the stride of the pyramid

    Returns:
        np.array: the reduced image
    """

    def compute_reduced_value(_image: np.array, reduced_image_x: int, reduced_image_y: int, _stride: int) -> int:
        start_x = reduced_image_x * _stride
        start_y = reduced_image_y * _stride
        end_x = start_x + _stride
        end_y = start_y + _stride

        for _ in range(start_x, end_x):
            for _ in range(start_y, end_y):
                if i > (_image.shape[0] - 1) or j > (_image.shape[1] - 1):
                    continue
                if _image[i][j] == 0:
                    return 0

        return 1

    # Allocate new image
    # Ceil has to be used because if the kernel does not cover perfectly the image ones will be added to the
    # original image to fit the kernel
    reduced_image_shape = [math.ceil(dim / stride) for dim in image.shape]
    reduced_image = np.zeros(reduced_image_shape, dtype=image.dtype)

    for i in range(reduced_image_shape[0]):
        for j in range(reduced_image_shape[1]):
            reduced_value = compute_reduced_value(image, i, j, stride)
            reduced_image[i][j] = reduced_value

    return reduced_image


def interpolate_dt_binary_image(dt_reduced_image: np.array, stride: int) -> np.array:
    """Interpolate the values of the distance transform image obtained from the reduced image

    The algorithm is sequential but it can be paralleled to obtain (hopefully) O(1) time complexity.

    Moreover the algorithm has been implemented giving priority to readability, sacrificing performance.
    If the results will be good a more efficient version of the algorithm will be implemented.

    Args:
        dt_reduced_image: the reduced distance transform image
        stride: the stride of the pyramid

    Returns:
        np.array: the interpolated distance transform image
    """

    dt_original_image_shape = [dim * stride for dim in dt_reduced_image.shape]
    dt_original_image = np.zeros(
        dt_original_image_shape, dtype=dt_reduced_image.dtype)

    # Initialization
    for i in range(dt_original_image.shape[0]):
        for j in range(dt_original_image.shape[1]):
            dt_original_image[i][j] = -1

    # 1) Fill original image with values obtained from the reduced one multiplied by stride
    #    The values will always be placed in the top left position
    #    Example
    """
         0 1
         1 2
         
         0 ? 2 ?
         ? ? ? ?
         2 ? 4 ?
         ? ? ? ?
    """
    for i in range(dt_reduced_image.shape[0]):
        for j in range(dt_reduced_image.shape[1]):
            dt_original_image[i*stride][j *
                                        stride] = dt_reduced_image[i][j] * stride

    # 2) Interpolate
    """
    Algorithm:
        For each not -1 value in the original image apply wave propagation considering a square
        centered in the value with radius/diagonal equals to (stride - 1)
        
        Square Example
        stride = 3
        
        x x x x x
        x x x x x
        x x 4 x x
        x x x x x
        x x x x x
        
        The x values indicated the pixels where wave propagation has to be applied
        
        Apply wave propagation this way collision between different centers will happen.
        This is the desired behaviour. In fact if for a pixel multiple values have been evaluated
        from different centers only the smallest one will be kept.
    
        This will ensure that a coherent dt image will be obtained, where coherent means that the distance between
        each two neighbours pixels is never greater than 1 and the growth direction is coherent.
        
        A detailed explanations and more examples can be found in my notebook.
    """

    for i in range(dt_reduced_image.shape[0]):
        for j in range(dt_reduced_image.shape[1]):
            original_image_center_position = (i * stride, j * stride)
            _wave_propagation_interpolation(
                dt_original_image, original_image_center_position, stride)

    return dt_original_image


def improved_interpolate_dt_binary_image(original_image: np.array, dt_reduced_image: np.array) -> np.array:
    """Improved version of the interpolate_dt_binary_image function

    Improvements:
        All the pixels equal to 0 in the original image (background/seeds)
        will be equal to 0 also in the interpolated image.
        Wave propagation will be executed also from all of this seeds.
        The algorithm should remain parallelize as the naive one.

    Changes:
        The stride parameter is no longer required because can be automatically computed from the shape
        of the images passed as parameters.

    Args:
        original_image: the original image
        dt_reduced_image: the reduced distance transform image

    Returns:
        np.array: the interpolated distance transform image
    """

    # Compute stride
    if original_image.shape[0] % dt_reduced_image.shape[0] != 0:
        raise Exception(f"stride is not int")
    stride = int(original_image.shape[0] / dt_reduced_image.shape[0])

    # Initialize to -1 except for background pixels
    dt_interpolated_image = np.zeros(
        original_image.shape, dtype=dt_reduced_image.dtype)
    for i in range(dt_interpolated_image.shape[0]):
        for j in range(dt_interpolated_image.shape[1]):
            if original_image[i][j] == 0:
                dt_interpolated_image[i][j] = 0
            else:
                dt_interpolated_image[i][j] = -1

    # Fill the interpolated image with values obtained from the reduced one (multiplied bu stride)
    # The background pixels will not be copied (the real background pixels have been copied from the original image)
    for i in range(dt_reduced_image.shape[0]):
        for j in range(dt_reduced_image.shape[1]):
            if dt_reduced_image[i][j] != 0:
                dt_interpolated_image[i * stride][j *
                                                  stride] = dt_reduced_image[i][j] * stride

    # Interpolate
    for i in range(dt_reduced_image.shape[0]):
        for j in range(dt_reduced_image.shape[1]):
            interpolated_image_center_position = (i * stride, j * stride)
            _wave_propagation_interpolation(
                dt_interpolated_image, interpolated_image_center_position, stride)

    return dt_interpolated_image


def _wave_propagation_interpolation(image: np.array, center_position: typing.Tuple[int, int], stride: int) -> None:
    """Wave propagation on an interpolated image.

    Precondition:
        The initial interpolated image (obtained before the interpolation starts), that is filled with the initial
        values obtained by the dt reduced image and, optionally, by the zeros obtained by the original image, can not
        have a zero in a cell that has also a non-zero value.
        This means that during the compression process the rule used tu compress the image is to always compress to a
        0 value if a the least a 0 value is present in the cell to compress.

        Example with stride 2\n
        not permitted\n
        0 x 2 x\n
        x 0 x 0\n
        The zero in the top left is not permitted since in the same cell a non-zero value is present.

    Args:
        image: The interpolated image.
        center_position: The position of the center of the square to apply wave propagation.
        stride: The stride of the image.
    """
    # Get sub-image centered in center_position and call wave propagation algorithm
    center_x = center_position[0]
    center_y = center_position[1]
    radius = stride - 1

    left_x = max(center_x - radius, 0)
    left_y = max(center_y - radius, 0)

    # +1 is necessary because right_x and right_y are included
    right_x = center_x + radius + 1
    right_y = center_y + radius + 1

    sub_image_center_x = center_x - left_x
    sub_image_center_y = center_y - left_y

    sub_image = image[left_x:right_x, left_y:right_y]
    # Get seeds
    seeds = []
    if sub_image[sub_image_center_x, sub_image_center_y] > 0:
        seeds.append((sub_image_center_x, sub_image_center_y))
    else:
        # If None the wave propagation algorithm will use all zeroes found in the image as seeds
        seeds = None

    output_image = wave_propagation_dt_image(sub_image, seeds=seeds)

    # Update the input image saving the minimum between the current value and the new one for each pixel
    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            original_x = i + left_x
            original_y = j + left_y
            if image[original_x][original_y] == -1:
                image[original_x][original_y] = output_image[i][j]
            else:
                image[original_x][original_y] = min(
                    image[original_x][original_y], output_image[i][j])
