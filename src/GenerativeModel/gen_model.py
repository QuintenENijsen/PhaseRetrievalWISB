import numpy as np
import numpy.typing as npt
from src.Initialization.pr_init import generate_gaussian_vector, generate_measurement_matrix, generate_measured, calculate_reconstruction_error, calculate_range
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import math
import os
import torch
from torch import nn
from src.ReconstructionAlgorithms.truncatedwf.truncatedwf import truncatedGradient
from src.Plotting.plotting import plot_heat_map_genmodel
from src.ReconstructionAlgorithms.truncatedwf.real_truncated_wf import new_trunc_spectral_init


device = torch.accelerator.current_accelerator().type

MAX_ITER: int = 300
alpha_fs = 3     #Paper states >= 3
alpha_f_lb = 0.3     #Paper states should be 0 <= alpha <= 0.5
alpha_f_ub = 5     #Paper states that >= 5

class G_Model(nn.Module):
    def __init__(self, d: int, n: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(d, n),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def G(model, X):
    if X.dim() == 1:
        X = X.unsqueeze(0)

    logits = model(X)
    return logits.squeeze(0)

def freeze_model(model):
    for param in model.Parameters():
        param.requires_grad = False

def z_step(model, z, f, lambda_, lr):
    #Sets all previously computed gradients to 0
    if z.grad is not None:
        z.grad.zero_()

    Gz = G(model, z)
    #print(Gz.grad_fn)
    loss = lambda_ * torch.norm(Gz - f.detach()) ** 2   #detach() may be wrong.
    #Update gradients of z
    loss.backward()
    with torch.no_grad():
        z -= lr * z.grad

    return z

def poisson_wirtinger_grad(A, f, y, alpha_fs):
    eps = 1e-10
    y = y.float()

    inner = A @ f
    intensities = inner ** 2

    a_norms = torch.norm(A, dim=1)
    f_norm = torch.norm(f)
    normalized_amp = (A.shape[1] ** 0.5 * inner.abs()) / (a_norms * f_norm + eps)

    mask1 = (normalized_amp >= alpha_f_lb) & (normalized_amp <= alpha_f_ub)

    residuals = (y - intensities).abs()
    K_t = residuals.mean()

    mask2 = residuals <= alpha_fs * K_t * normalized_amp

    mask = (mask1 & mask2).float()

    valid = intensities > eps
    weights = torch.where(valid, (1.0 - y / (intensities + eps)) * inner,
                          torch.zeros_like(inner))
    weights = weights * mask.float()

    return A.T @ weights

def optimize_model(d: int, n: int, m: int, A: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    model = G_Model(d,n).to(device)
    #freeze_model(model)
    mu = 0.2
    lambda_ = 1
    stepsize = mu / m

    f = torch.from_numpy(new_trunc_spectral_init(A, y, n, m, alpha_fs, True)).to(device)
    z = torch.randn(d, requires_grad = True, device=device) #Zoek hier nog een daadwerkelijk goeie initializer voor.
    A = torch.from_numpy(A).to(device)
    y = torch.from_numpy(y).to(device)
    for _ in range(0, MAX_ITER):
        #We first do one iteration of optimzation for f, with fixed z.
        #For this we use Truncated wirtinger flow.
        grad_f_torch = poisson_wirtinger_grad(A, f, y, alpha_fs) + 2 * (f - G(model, z))

        f = f - stepsize * grad_f_torch
        z = z_step(model, z, f, lambda_, 0.1)

    return f.detach().cpu().numpy()

def calc_reconstruction_error(inputs) -> (int, int, float):
    n = 64
    d = inputs[0]
    oversampling = inputs[1]
    m = oversampling * n


    ground_truth = generate_gaussian_vector(n) * 10
    measurement_maps = [generate_measurement_matrix(n, m) for _ in range(0, 30)]
    measurements = [generate_measured(M, ground_truth, m) for M in measurement_maps]

    estimators = [optimize_model(d, n, m, A, y) for A, y in zip(measurement_maps, measurements)]
    errors = list(map(lambda estimate: calculate_reconstruction_error(estimate, ground_truth), estimators))

    print(str(inputs) + " , Completed")
    average = sum(errors) / (10 * len(errors))

    return d, oversampling, average


def run_simulation():
     neural_net_dim = [8, 16, 32, 48]
     oversampling = [1, 2, 3, 4, 5, 6, 8, 10]

     jobs = list(product(neural_net_dim, oversampling))

     all_results = list(map(calc_reconstruction_error, jobs))
     #with ProcessPoolExecutor() as executor:
     #    all_results = list(executor.map(calc_reconstruction_error, jobs))

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

if __name__ == "__main__":
    run_simulation()