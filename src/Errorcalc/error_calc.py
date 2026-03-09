import sys
import numpy as np
from numpy import linalg as la
import numpy.typing as npt
from src.Initialization.pr_init import calculate_measurement

def calculate_reconstruction_error(f: npt.NDArray[np.float64], f0: npt.NDArray[np.float64]) -> float:
    return min(float(la.norm(f - f0)), float(la.norm(f + f0)))

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