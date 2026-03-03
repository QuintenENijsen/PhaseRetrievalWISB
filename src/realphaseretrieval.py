import sys

import numpy as np
from numpy import linalg as la
import numpy.typing as npt
import random as rand
import math
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from plotting import plot_heat_map
from numba import njit

SEED_GROUND_TRUTH = ""
SEED_MEASUREMENT_MAP = ""


#Dimension of ground_truth
#n: int = 5
#Dimension of measurement
#m: int = 200*n
#Stopping condition wirtinger flow
eps = 1e-05
MAX_ITER = 10_000
#The norm of the ground truth, we want to be able to control this to differentiate high and low energy regimes
norm_f0 = 100

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

def calculate_measurement(matrix: npt.NDArray[np.float64], row: int, f: npt.NDArray[np.float64], m: int) -> float:
    if row >= m:
        return -1
    else:
        a_row = matrix[row]
        return abs(np.inner(a_row, f)) ** 2

def generate_measured(matrix: npt.NDArray[np.float64], f0: npt.NDArray[np.float64], m: int) -> npt.NDArray[np.float64]:
    """Generates the y in the poisson phase retrieval problem, using the matrix A and assuming mathcal{A} = |Af|^2"""
    intensities = (np.dot(matrix, f0))**2
    return np.array([np.random.poisson(x) for x in intensities])

@njit
def spectral_initialization(y: npt.NDArray[np.float64], A: npt.NDArray[np.float64], n: int, m: int) -> npt.NDArray[np.float64]:
    """
    Performs initialization based on the spectral initialization method as defined in Candès, et al. (2015)
    :param y: The measurements
    :param A: The measurement matrix
    :return: The Eigenvector of the largest Eigenvalue of 1/m sum_{i=1}^m y_i a_i a_i^t
    """
    #Calculate the matrix 1/m \sum_{i=1}^m y_i a_i a_i^t
    Y: npt.NDArray[np.float32] = np.zeros((n, n), dtype=np.float64)

    for i in range (0, n):
        a_i = np.ascontiguousarray(A[i])
        Y = Y + y[i] * (np.outer(a_i ,a_i))

    Y = Y / m
    #Get the eigenvector
    eigenvalues, eigenvectors = la.eigh(Y)
    max_index = np.argmax(eigenvalues)

    new_norm = math.sqrt(n * np.sum(y.astype(np.float64)) / np.sum(A**2))
    max_eigenvector = np.ascontiguousarray(eigenvectors[:, max_index]).reshape(n)

    return (new_norm / np.dot(max_eigenvector, max_eigenvector)) * max_eigenvector

@njit
def compute_wirtinger_gradient(f: npt.NDArray[np.float64], A: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    inner_products = np.dot(A, f)
    intensities = inner_products ** 2
    weights = 1.0 - y / (intensities + 1e-10) # + 1e-10 to not get division by zero errors whenever there are zero intensity measurements.
    weighted = weights * inner_products
    return np.dot(A.T, weighted)

@njit
def find_minimizer(A: npt.NDArray[np.float64], y: npt.NDArray[np.float64], n: int, m: int) -> npt.NDArray[np.float64]:
    """
    Performs the wirtinger flow algorithm to reconstruct the ground truth signal. Assumes the Poisson MLE formulation of the phase retrieval problem.
    :param A: The measurement matrix A, with mathcal{A}(f) = |Af|^2 the measurement map
    :param f0: The ground truth signal
    :param y: The intensity measurement of our signal f_0. y_i distributed Pois(|<a_i, f_0>|^2)
    :return:
    """
    f = spectral_initialization(y, A, n, m)
    #Step size for the update function
    mu: float = 1 / pow(la.norm(A), 2)
    ix = 0
    while ix < MAX_ITER:
        grad = compute_wirtinger_gradient(f, A, y)
        f = f - (mu / n) * compute_wirtinger_gradient(f, A, y)
        ix += 1
        if np.linalg.norm(grad) < eps:
            break
    return f

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

def compute_min_errors_ranges(measurement_map, measurement, n, m, ground_truth):
    minimizer = find_minimizer(measurement_map, measurement, n, m)
    error = calculate_reconstruction_error(minimizer, ground_truth)
    range_ = calculate_range(measurement_map, ground_truth, minimizer, m)
    return minimizer, error, range_

def compute_for_nm(n_m, norm_f0):
    n = n_m[0]
    m_ratio = n_m[1]

    m = m_ratio * n
    ground_truth = generate_gaussian_vector(n) * norm_f0
    measurement_maps = [generate_measurement_matrix(n, m) for _ in range(1, 50)]
    measurements = [generate_measured(M, ground_truth, m) for M in measurement_maps]

    results = [compute_min_errors_ranges(mm, meas, n, m, ground_truth)
               for mm, meas in zip(measurement_maps, measurements)]

    minimizers, errors, ranges = zip(*results)
    minrange, maxrange = zip(*ranges)

    avg_error = sum(errors) / len(errors)
    avg_maxrange = min(50, sum(maxrange) / len(maxrange))

    print(str(n) + ", " + str(m_ratio) + " completed.")
    return n, m_ratio, avg_error, avg_maxrange

def run_average_sim():
    ns = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]# 45, 50]
    oversampling_ratios = [2 * n**2 for n in ns]  # your ms list

    jobs = list(product(ns, oversampling_ratios))

    with ProcessPoolExecutor() as executor:
        all_results = list(executor.map(compute_for_nm, jobs, [norm_f0 for x in range(len(jobs))]))

    # Reconstruct per-n lists
    print(all_results)

    errors_across_n = {n: [] for n in ns}
    maxranges_across_n = {n: [] for n in ns}

    for n, m_ratio, avg_error, avg_maxrange in all_results:
        errors_across_n[n].append(avg_error)
        maxranges_across_n[n].append(avg_maxrange)

    # Lookup dictionary
    error_lookup = {(n, m): err for n, m, err, _ in all_results}

    # Build matrix
    error_matrix = np.array([
        [error_lookup[(n, m)] for m in oversampling_ratios]
        for n in ns
    ])

    print(error_matrix)

    #Here the heat map picture function should come.
    plot_heat_map(ns, oversampling_ratios, error_matrix, norm_f0)

if __name__ == "__main__":
    run_average_sim()