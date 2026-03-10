import sys
import numpy as np
from numpy import linalg as la
import numpy.typing as npt
import random as rand
import math
from numba import njit


@njit
def vector_norm(f: npt.NDArray[np.float64]) -> float:
    return math.sqrt(np.sum(np.square(f)))

def generate_gaussian_vector(n: int) -> npt.NDArray[np.float64]:
    """
    :return: A vector of size n whose components are Gaussian distributed random variables, where the vector has norm 1 in the ell^2 norm.
    """

    rand_vec: npt.NDArray[np.float64] = np.array([rand.gauss(0, 1) for x in range(0, n)], dtype=np.float64)
    #Normalize the vector
    norm_vec = la.norm(rand_vec)

    return rand_vec * (1.0 / norm_vec)

def generate_measurement_matrix(n: int, m: int) -> npt.NDArray[np.float64]:
    return np.array([generate_gaussian_vector(n) for x in range(0, m)])

def generate_measured(matrix: npt.NDArray[np.float64], f0: npt.NDArray[np.float64], m: int) -> npt.NDArray[np.float64]:
    """Generates the y in the poisson phase retrieval problem, using the matrix A and assuming mathcal{A} = |Af|^2"""
    intensities = (np.dot(matrix, f0))**2
    return np.array([np.random.poisson(x) for x in intensities])

def calculate_measurement(matrix: npt.NDArray[np.float64], row: int, f: npt.NDArray[np.float64], m: int) -> float:
    if row >= m:
        return -1
    else:
        a_row = matrix[row]
        return abs(np.inner(a_row, f)) ** 2

def calculate_reconstruction_error(f: npt.NDArray[np.float64], f0: npt.NDArray[np.float64]) -> float:
    return min(vector_norm(f - f0), vector_norm(f + f0))

def calculate_range(matrix: npt.NDArray[np.float64], f0: npt.NDArray[np.float64], f: npt.NDArray[np.float64], m: int) -> type[float, float]:
    """
    :return: The range that the fraction (|<a_i, tilde{f}>|^2 / |<a_i, f_0>|^2) lives in. The left float is the minimum, right the maximum.
    """
    min_val: float = sys.float_info.max
    max_val: float = sys.float_info.min

    for i in range(0, m):
        measurement_f: float = calculate_measurement(matrix, i, f, m)
        measurement_f0: float = calculate_measurement(matrix, i, f0, m)
        #print("f measurement: " + str(measurement_f) + "; f0 measurement: " + str(measurement_f0))

        quotient = measurement_f / measurement_f0
        #if quotient > 1e5:
            #print(str(f) + "; " + str(f0))
        if quotient < min_val:
            min_val = quotient
        if quotient > max_val:
            max_val = quotient

    return min_val, max_val
