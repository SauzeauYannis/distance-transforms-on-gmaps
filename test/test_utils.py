import numpy as np


def matrix_compare(expected_matrix: np.array, actual_matrix: np.array) -> True:
    """
    It returns True if the two matrix are equal.
    It returns False otherwise.
    """
    for i in range(expected_matrix.shape[0]):
        for j in range(expected_matrix.shape[1]):
            if expected_matrix[i][j] != actual_matrix[i][j]:
                return False

    return True