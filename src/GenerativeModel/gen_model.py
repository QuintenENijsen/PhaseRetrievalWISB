#Pytorch imports
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

#Locally defined modules
from src.ReconstructionAlgorithms.truncatedwf.truncatedwf import truncatedGradient
from src.Plotting.plotting import plot_heat_map_genmodel
from src.ReconstructionAlgorithms.truncatedwf.real_truncated_wf import new_trunc_spectral_init
from src.Initialization.pr_init import generate_gaussian_vector, generate_measurement_matrix, generate_measured, calculate_reconstruction_error, calculate_range

#Other imports
import numpy as np
import numpy.typing as npt
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import math
import os

#GPU detection.
device = torch.accelerator.current_accelerator().type

MAX_ITER: int = 300
alpha_fs = 3     #Paper states >= 3
alpha_f_lb = 0.3     #Paper states should be 0 <= alpha <= 0.5
alpha_f_ub = 5     #Paper states that >= 5

#######################################################################################################################################
#Defining the generative model plus accessory functions
#######################################################################################################################################
class G_Model(nn.Module):
    def __init__(self, d: int, n: int):
        super().__init__()
        self.flatten = nn.Flatten()
        #hidden = max(n, 2 * d)
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
    for param in model.parameters():
        param.requires_grad = False


#######################################################################################################################################
# Data loading
#######################################################################################################################################

training_size = 10000
validation_size = 100

training_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

training_data = torch.randperm(len(training_dataset))[:training_size]

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

test_data = torch.randperm(len(test_dataset))[:validation_size]

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

def flatten_data(dataloader):
    X = []
    for images, _ in dataloader:
        images = images.view(images.size(0), -1)
        X.append(images)
    return torch.cat(X, dim=0)

def compute_pca(X, n, d):
    """
    Computes the PCA matrix for data matrix X, with embedding dimension k.
    :param X: Data list
    :param k: Embedding dimension
    :param d: Latent dimension
    :return: A tuple with the bias and the PCA matrix
    """
    mu = X.mean(dim=0, keepdim=True)
    X_centered = X - mu

    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

    components = Vt[:n]
    return mu, components

def G_pca(mu, U, z):
    return mu + U @ z

#######################################################################################################################################
#Calculating reconstruction error
#######################################################################################################################################

def z_step(model, z, f, lambda_, lr):
    #Sets all previously computed gradients to 0
    if z.grad is not None:
        z.grad.zero_()

    Gz = G(model, z)
    #print(Gz.grad_fn)
    loss = torch.norm(Gz - f.detach()) ** 2   #detach() may be wrong.
    #Update gradients of z
    loss.backward()
    with torch.no_grad():
        z -= lr * lambda_ * z.grad

    return z.detach().requires_grad_(True)

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

def optimize_model(d: int, n: int, m: int, A: npt.NDArray[np.float64], y: npt.NDArray[np.float64], model) -> npt.NDArray[np.float64]:
    #model = G_Model(d,n).to(device)
    #freeze_model(model)
    mu = 0.2
    lambda_ = 5 * math.sqrt(d)
    stepsize = mu / (lambda_ * m)
    z_lr = 0.1 / lambda_

    f = torch.from_numpy(new_trunc_spectral_init(A, y, n, m, alpha_fs, True)).to(device)
    z = torch.randn(d, requires_grad = True, device=device) #Zoek hier nog een daadwerkelijk goeie initializer voor.

    A = torch.from_numpy(A).to(device)
    y = torch.from_numpy(y).to(device)
    for _ in range(0, MAX_ITER):
        #We first do one iteration of optimzation for f, with fixed z.
        #For this we use Truncated wirtinger flow.
        grad_f_torch = poisson_wirtinger_grad(A, f, y, alpha_fs) + 2 * lambda_ * (f - G(model, z))

        f = f - stepsize * grad_f_torch
        z = z_step(model, z, f, lambda_, z_lr)

    return f.detach().cpu().numpy()

def calc_reconstruction_error(inputs) -> (int, int, float):
    n = 128
    d = inputs[0][0]
    model = inputs[0][1]
    model.to(device)
    freeze_model(model)
    oversampling = inputs[1]
    m = oversampling * n

    ground_truth = generate_gaussian_vector(n) * 100
    measurement_maps = [generate_measurement_matrix(n, m) for _ in range(0, 50)]
    measurements = [generate_measured(M, ground_truth, m) for M in measurement_maps]

    estimators = [optimize_model(d, n, m, A, y, model) for A, y in zip(measurement_maps, measurements)]
    errors = list(map(lambda estimate: calculate_reconstruction_error(estimate, ground_truth), estimators))

    average = sum(errors) / (100 * len(errors))

    print(str(inputs[0][0]) +"," + str(inputs[1]) + " , Completed " + ", Average: " + str(average))
    return d, oversampling, average


def run_simulation():
     neural_net_dim = [8, 16, 32, 48, 64, 80, 96, 112, 128]
     models = [G_Model(d, 128) for d in neural_net_dim]
     oversampling = [1, 2, 3, 4, 5, 6, 7, 8]

     jobs = list(product(zip(neural_net_dim, models), oversampling))

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