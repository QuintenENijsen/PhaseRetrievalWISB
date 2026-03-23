import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # FIX 1: lets Python find truncatedwf.so

from src.Initialization.pr_init import generate_gaussian_vector, generate_measurement_matrix, generate_measured, calculate_reconstruction_error, calculate_range
import numpy as np
from numpy import linalg as la
import numpy.typing as npt
import math
from src.Plotting.plotting import plot_heat_map_norm, plot_heat_map_truncrate
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from src.ReconstructionAlgorithms.truncatedwf.truncatedwf import truncGradientDescent, truncCountingSpectralInit

import cProfile
import pstats

eps = 1e-30
MAX_ITER = 15_000
norm_f0 = 100

#Fake values
alpha_y = 3     #Paper states >= 3
alpha_f_lb = 0.3     #Paper states should be 0 <= alpha <= 0.5
alpha_f_ub = 5     #Paper states that >= 5

def trunc_spectral_init(A: npt.NDArray[np.float64],y: npt.NDArray[np.float64], n: int, m: int, trunc: bool)-> npt.NDArray[np.float64]:
    lambda_0: float = (np.sum(y.astype(np.float64))) / m
    factor: float = math.sqrt(m * n / (np.sum(np.linalg.norm(A, axis=1))))

    Y: npt.NDArray[np.float32] = np.zeros((n, n), dtype=np.float64)

    for i in range (0, n):
        if trunc and abs(y[i]) > alpha_y**2 * lambda_0**2:
            continue
        a_i = np.ascontiguousarray(A[i])
        Y = Y + y[i] * (np.outer(a_i ,a_i))

    Y = Y / m
    eigenvalues, eigenvectors = la.eigh(Y)
    max_index = np.argmax(eigenvalues)

    max_eigenvector = np.ascontiguousarray(eigenvectors[:, max_index]).reshape(n)

    return factor * lambda_0 * max_eigenvector

def counting_trunc_spectral_init(y: npt.NDArray[np.float64], n: int, m: int) -> tuple[int, int]:
    lambda_0: float = (np.sum(y.astype(np.float64))) / m

    truncated_count = 0
    non_zero_count = 0
    for i in range (0, n):
        if y[i] > 0:
            non_zero_count += 1
            if abs(y[i]) > alpha_y**2 * lambda_0**2:
                truncated_count += 1


    return truncated_count, non_zero_count

def gradient_descent(A: npt.NDArray[np.float64], y: npt.NDArray[np.float64], n: int, m: int):
    f = trunc_spectral_init(A, y, n, m, False)
    mu = 0.2  #Chosen based on the paper stating that we should have 0 < mu < 0.28.
    minimizer =  truncGradientDescent(f, y, A, mu, MAX_ITER, eps, alpha_f_lb, alpha_f_ub, alpha_y)
    return minimizer

def compute_min_errors_ranges(measurement_map, measurement, n, m, ground_truth):
    minimizer = gradient_descent(measurement_map, measurement, n, m)
    error = calculate_reconstruction_error(minimizer, ground_truth)
    #range_ = calculate_range(measurement_map, ground_truth, minimizer, m)
    return minimizer, error, (0,0)

def compute_for_nm(norm_oversampling):
    n = 32
    #n = n_m[0]
    m_ratio = norm_oversampling[1]

    m = m_ratio * n
    ground_truth = generate_gaussian_vector(n) * norm_oversampling[0]
    measurement_maps = [generate_measurement_matrix(n, m) for _ in range(0, 30)]
    measurements = [generate_measured(M, ground_truth, m) for M in measurement_maps]

    results = [compute_min_errors_ranges(mm, meas, n, m, ground_truth)
               for mm, meas in zip(measurement_maps, measurements)]

    minimizers, errors, ranges = zip(*results)
    minrange, maxrange = zip(*ranges)

    avg_error = sum(errors) / ((norm_oversampling[0]) * len(errors))     #Only relative error
    avg_maxrange = min(50, sum(maxrange) / len(maxrange))

    print(str(norm_oversampling[0]) + ", " + str(m_ratio) + " completed.")
    return norm_oversampling[0], m_ratio, avg_error, avg_maxrange

def run_average_sim():
    #ns = [5, 10, 15, 20]# 45, 50]
    norms = [1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1, 1]
    oversampling_ratios = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 30]  # your ms list

    jobs = list(product(norms, oversampling_ratios))

    with ProcessPoolExecutor() as executor:
        all_results = list(executor.map(compute_for_nm, jobs))
    print(all_results)

    errors_across_n = {n: [] for n in norms}
    #maxranges_across_n = {n: [] for n in norms}

    for n, m_ratio, avg_error, avg_maxrange in all_results:
        errors_across_n[n].append(avg_error)
        #maxranges_across_n[n].append(avg_maxrange)

    error_lookup = {(n, m): err for n, m, err, _ in all_results}

    error_matrix = np.array([
        [error_lookup[(n, m)] for m in oversampling_ratios]
        for n in norms
    ])

    print(error_matrix)

    plot_heat_map_norm(norms, oversampling_ratios, error_matrix, norms[len(norms)-1])

def truncation_spectral_init(norms_alpha):
    n = 32
    oversampling_ratio = 10
    m = oversampling_ratio * n

    norm = norms_alpha[0]
    alpha_f = norms_alpha[1]

    ground_truth = generate_gaussian_vector(n) * norm
    measurement_maps = [generate_measurement_matrix(n, m) for _ in range(0,50)]
    measurements = [generate_measured(M, ground_truth, m) for M in measurement_maps]

    results = [truncCountingSpectralInit(measured, alpha_f, n, m) for measured in measurements]

    trunc_count, non_zero_count = zip(*results)
    avg_trunc_count = sum(trunc_count) / len(trunc_count)
    avg_non_zero_count = sum(non_zero_count) / len(non_zero_count)
    avg_trunc_rate = avg_trunc_count / avg_non_zero_count

    return norm, alpha_f, avg_trunc_rate

def find_init_truncation_rate():
    norms = [5e-1, 6e-1, 7e-1, 8e-1, 9e-1, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
    alpha_fs = [3, 4, 5, 6, 7, 8, 9, 10]

    jobs = list(product(norms, alpha_fs))

    with ProcessPoolExecutor() as executor:
        all_results = list(executor.map(truncation_spectral_init, jobs))

    print(all_results)

    count_across_norm = {n: [] for n in norms}

    for norm, alpha_f, rate in all_results:
        count_across_norm[norm].append(rate)

    rate_lookup = {(norm, alpha_f): rate for norm, alpha_f, rate in all_results}

    error_matrix = np.array([
        [rate_lookup[(n, m)] for m in alpha_fs]
        for n in norms
    ])

    plot_heat_map_truncrate(norms, alpha_fs, error_matrix)

#profiler = cProfile.Profile()
#profiler.enable()
#run_average_sim()
#profiler.disable()

#stats = pstats.Stats(profiler)
#stats.sort_stats('cumulative')
#stats.print_stats(20)

if __name__ == "__main__":
    run_average_sim()