from src.Initialization.pr_init import generate_gaussian_vector, generate_measurement_matrix, generate_measured, calculate_reconstruction_error, calculate_range
import numpy as np
from numpy import linalg as la
import numpy.typing as npt
import math
from numba import njit
from src.Plotting.plotting import plot_heat_map
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import truncatedwf


eps = 1e-05
MAX_ITER = 10_000
norm_f0 = 100

#Fake values
alpha_y = 10
alpha_f_lb = 10
alpha_f_ub = 10

@njit
def trunc_spectral_init(A: npt.NDArray[np.float64],y: npt.NDArray[np.float64], n: int, m: int)-> npt.NDArray[np.float64]:
    lambda_0: float = (np.sum(y.astype(np.float64))) / m
    factor: float = math.sqrt(m * n / (np.sum(np.linalg.norm(A, axis=1))))

    Y: npt.NDArray[np.float32] = np.zeros((n, n), dtype=np.float64)

    for i in range (0, n):
        if abs(y[i]) > alpha_y**2 * lambda_0**2:
            continue
        a_i = np.ascontiguousarray(A[i])
        Y = Y + y[i] * (np.outer(a_i ,a_i))

    Y = Y / m
    eigenvalues, eigenvectors = la.eigh(Y)
    max_index = np.argmax(eigenvalues)

    max_eigenvector = np.ascontiguousarray(eigenvectors[:, max_index]).reshape(n)

    return factor * lambda_0 * max_eigenvector

@njit
def gradient_descent(A: npt.NDArray[np.float64], y: npt.NDArray[np.float64], n: int, m: int):
    f = trunc_spectral_init(A, y, n, m)
    mu = 0  #Make this some suitable positive constant, should be in the paper.
    ix = 0
    while ix < MAX_ITER:
        grad = truncatedwf.truncatedGradient(f, y, A, n)
        f =  f - (mu / m) * grad
        ix += 1
        if la.norm(grad) < eps:
            break

    return f

def compute_min_errors_ranges(measurement_map, measurement, n, m, ground_truth):
    minimizer = gradient_descent(measurement_map, measurement, n, m)
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
    ns = [5, 7, 10, 12, 15, 17, 20, 22, 25, 27]# 45, 50]
    oversampling_ratios = [20 * n for n in ns]  # your ms list

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
