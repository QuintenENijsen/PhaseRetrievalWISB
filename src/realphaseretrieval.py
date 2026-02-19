import sys

import numpy as np
from numpy import linalg as la
import numpy.typing as npt
import random as rand
import math

SEED_GROUND_TRUTH = ""
SEED_MEASUREMENT_MAP = ""


#Dimension of ground_truth
n: int = 20
#Dimension of measurement
m: int = n
#Stopping condition wirtinger flow
eps = 1e-06

def vector_norm(f: npt.NDArray[np.float32]) -> float:
    return math.sqrt(np.sum(np.square(f)))

def generate_gaussian_vector() -> npt.NDArray[np.float32]:
    """
    :return: A vector of size n whose components are Gaussian distributed random variables, where the vector has norm 1 in the ell^2 norm.
    """

    rand_vec: npt.NDArray[np.float32] = np.array([rand.gauss(0, 1) for x in range(1, n)], dtype=np.float32)
    #Normalize the vector
    norm_vec: float = vector_norm(rand_vec)

    return rand_vec * (1.0 / norm_vec)

def generate_measurement_matrix() -> npt.NDArray[np.float32]:
    return np.array([generate_gaussian_vector() for x in range(1, m)])


def calculate_measurement(matrix: npt.NDArray[np.float32], row: int, f: npt.NDArray[np.float32]) -> float:
    if row > m:
        return -1
    else:
        a_row = matrix[row]
        return abs(np.inner(a_row, f)) ** 2

def generate_measured(matrix: npt.NDArray[np.float32], f0: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Generates the y in the poisson phase retrieval problem, using the matrix A and assuming \mathcal{A} = |Af|^2"""
    y = []

    for i in range (1, m):
        y.append(np.random.poisson(calculate_measurement(matrix, i, f0)))

    return np.array(y)

def spectral_initialization(y: npt.NDArray[np.float32], A: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Performs initialization based on the spectral initialization method as defined in Candès, et al. (2015)
    :param y: The measurements
    :param A: The measurement matrix
    :return: The Eigenvector of the largest Eigenvalue of 1/m \sum_{i=1}^m y_i a_i a_i^t
    """
    #Calculate the matrix 1/m \sum_{i=1}^m y_i a_i a_i^t
    Y: npt.NDArray[np.float32] = np.zeros((n, n), dtype=np.float32)

    for i in range (1, n):
        Y = Y + y[i] * (np.outer(A[i], A[i]))

    Y = 1/m * Y
    #Get the eigenvector
    eigenvalues, eigenvectors = la.eig(Y)
    max_index = np.argmax(eigenvalues)

    return eigenvectors[max_index]

def compute_wirtinger_gradient(f: npt.NDArray[np.float32], A: npt.NDArray[np.float32], y: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    inner_products = A @ f
    intensities = inner_products ** 2
    weights = 1.0 - y / (intensities + 1e-10) # + 1e-10 to not get division by zero errors whenever there are zero intensity measurements.
    weighted = weights * inner_products
    return A.T @ weighted

def find_minimizer(A: npt.NDArray[np.float32], y: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Performs the wirtinger flow algorithm to reconstruct the ground truth signal. Assumes the Poisson MLE formulation of the phase retrieval problem.
    :param A: The measurement matrix A, with \mathcal{A}(f) = |Af|^2 the measurement map
    :param f0: The ground truth signal
    :param y: The intensity measurement of our signal f_0. y_i distributed Pois(|<a_i, f_0>|^2)
    :return:
    """
    f = spectral_initialization(y, A)
    #Step size for the update function
    mu: float = 1 / pow(la.norm(A), 2)

    while True:
        grad = compute_wirtinger_gradient(f, A, y)
        f = f - (mu / n) * compute_wirtinger_gradient(f, A, y)

        if np.linalg.norm(grad) < eps:
            break

    return f

def calculate_reconstruction_error(f: npt.NDArray[np.float32], f0: npt.NDArray[np.float32]) -> float:
    return min(vector_norm(f - f0), vector_norm(f + f0))

def calculate_range(matrix: npt.NDArray[np.float32], f0: npt.NDArray[np.float32], f: npt.NDArray[np.float32]) -> type[float, float]:
    """
    :return: The range that the fraction (|<a_i, tilde{f}>|^2 / |<a_i, f_0>|^2) lives in. The left float is the minimum, right the maximum.
    """
    min_val: float = sys.float_info.max
    max_val: float = sys.float_info.min

    for i in range(1, m):
        measurement_f: float = calculate_measurement(matrix, i, f)
        measurement_f0: float = calculate_measurement(matrix, i, f0)

        quotient = measurement_f / measurement_f0
        if quotient > min_val:
            min_val = quotient
        if quotient > max_val:
            max_val = quotient

    return min_val, max_val


def run_phase_retrieval():
    ground_truth = generate_gaussian_vector()
    measurement_maps = [generate_measurement_matrix() for x in range(1, 100)]
    measurements = map(lambda M: generate_measured(M, ground_truth), measurement_maps)
    minimizers = map(lambda elem: find_minimizer(elem[0],elem[1]) , zip(measurement_maps, measurements))

    #Calculate reconstruction errors and ranges
    errors = map(lambda f: calculate_reconstruction_error(f, ground_truth),minimizers)
    ranges = map(lambda mapsmins: calculate_range(mapsmins[0], ground_truth, mapsmins[1]),zip(measurement_maps, minimizers))