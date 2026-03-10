from src.Initialization.pr_init import generate_gaussian_vector, generate_measurement_matrix, generate_measured
import numpy as np
from numpy import linalg as la
import numpy.typing as npt
import math
from numba import njit
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

def compute_for_nm():
    pass

def run_average_sim():
    pass