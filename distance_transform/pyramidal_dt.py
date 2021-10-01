import numpy as np
import math


from wave_propagation import wave_propagation_dt_binary_image


def pyramidal_dt_binary_image(image: np.array, stride: int) -> np.array:
    """
    Parallel approximate algorithm that should (hopefully) takes O(log(n))
    where n is the number of darts

    Time complexity: O(log(n)) ?
    Space complexity: O(n) ?

    Algorithm:
    1) Reduce image size through a pyramid
    2) Compute dt on a reduced image
    3) Interpolate the obtained values for each level

    :param image:
    :param stride:
    :return:
    """

    dt_image = np.zeros(image.shape, dtype=image.dtype)

    # reduce image
    reduced_image = reduce_size_binary_image(image, stride)

    # compute dt
    dt_image = wave_propagation_dt_binary_image(reduced_image)

    # interpolate


    return dt_image


def reduce_size_binary_image(image: np.array, stride: int) -> np.array:
    """
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

    :param image:
    :param stride: for simplicity for the moment it is only an integer (square kernel). It could be a tuple of integers
                   if kernels of different forms would be used
    :return:
    """

    def compute_reduced_value(image: np.array, reduced_image_x: int, reduced_image_y: int, stride: int) -> int:
        start_x = reduced_image_x * stride
        start_y = reduced_image_y * stride
        end_x = start_x + stride
        end_y = start_y + stride

        for i in range(start_x, end_x):
            for j in range(start_y, end_y):
                if i > (image.shape[0] - 1) or j > (image.shape[1] - 1):
                    continue
                if image[i][j] == 0:
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
    """
    PRECONDITIONS:
    1) An odd stride has to be used

    :param dt_reduced_image:
    :param stride:
    :return:
    """

    if stride % 2 == 0:
        raise Exception("The stride has to be odd")

    dt_original_image_shape = [dim * stride for dim in dt_reduced_image.shape]
    dt_original_image = np.zeros(dt_original_image_shape, dtype=dt_reduced_image.dtype)

    # Not so easy. I should do it tomorrow morning, with fresh energies.
    # 1) I can compute the average? But what does it mean? It doesn't have any sense
    # 2) My idea is to increase or decrease by 1 the distance moving from the center of the interpolated cell
    #    I have to compute the direction of growth in order to understand if I have to increase or decrease the distance
    #    View photo saved on Telegram for details on that.