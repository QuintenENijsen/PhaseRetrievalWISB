import numpy as np
import numpy.typing as npt
from src.Initialization.pr_init import generate_gaussian_vector, generate_measurement_matrix, generate_measured, calculate_reconstruction_error, calculate_range
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import math

from src.Plotting.plotting import plot_heat_map_genmodel


def G(x: npt.NDArray[np.float64]):
    """
    Defines the geneative model used during reconstruction
    :param x: The input vector of the generative model, lives in \R^d
    :return: The value G(x) \in \R^n
    """
    pass



def optimize_model() -> npt.NDArray[np.float64]:

    pass

def calc_reconstruction_error(inputs):
    n = 24
    k = inputs[0]
    oversampling = inputs[1]
    m = oversampling * n


    ground_truth = generate_gaussian_vector(n) * 10
    measurement_maps = [generate_measurement_matrix(n, m) for _ in range(0,100)]
    measurements = [generate_measured(M, ground_truth, m) for M in measurement_maps]

    estimators = [optimize_model() for A, y in zip(measurement_maps, measurements)]
    errors = list(map(lambda estimate: calculate_reconstruction_error(estimate, ground_truth), estimators))

    print(str(inputs) + " , Completed")
    average = sum(errors) / (10 * len(errors))
    return k, oversampling, average


 def run_simulation():
     neural_net_dim = [5]
     oversampling = [2, 4, 6, 8, 10, 15, 20, 25]

     jobs = list(product(neural_net_dim, oversampling))

     with ProcessPoolExecutor() as executor:
         all_results = list(executor.map(calc_reconstruction_error, jobs))

     print(all_results)

     count_across_norm = {n: [] for n in neural_net_dim}

     for norm, alpha_f, rate in all_results:
         count_across_norm[norm].append(rate)

     rate_lookup = {(norm, alpha_f): error for norm, alpha_f, error in all_results}

     error_matrix = np.array([
         [rate_lookup[(n, m)] for m in oversampling]
         for n in neural_net_dim
     ])

     plot_heat_map_genmodel(neural_net_dim, oversampling, error_matrix, 1)

     pass