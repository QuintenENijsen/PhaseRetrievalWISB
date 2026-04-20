#Pytorch imports
from importlib.metadata import requires

import torch
from torch import nn, optim
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

MAX_ITER: int = 1000
alpha_fs = 3     #Paper states >= 3
alpha_f_lb = 0.3     #Paper states should be 0 <= alpha <= 0.5
alpha_f_ub = 5     #Paper states that >= 5

validation_batch_size = 20

#######################################################################################################################################
#Small utility functions and initialization
#######################################################################################################################################

def torch_recon_error(f, f_0):
    return torch.minimum(torch.norm(f-f_0, dim=-1), torch.norm(f+f_0, dim=-1))

def generate_measurement_matrix_gpu(n: int, m: int , k:int =1):
    return torch.randn(k, m, n, device=device, dtype=torch.float32)

def generate_measurement_gpu(A, f_0):
    inner = A @ f_0
    intensities = inner ** 2
    return torch.poisson(intensities)

#######################################################################################################################################
# Data loading
#######################################################################################################################################

training_size = 100000
validation_size = 10

training_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

training_indices = torch.randperm(len(training_dataset))[:training_size]
training_data = Subset(training_dataset, training_indices)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

test_indices = torch.randperm(len(test_dataset))[:validation_size]
test_data = Subset(test_dataset, test_indices)

train_dataloader = DataLoader(training_data,
                              batch_size=128,
                              num_workers=6,
                              pin_memory=True,
                              persistent_workers=True,
                              prefetch_factor=4,
                              )
test_dataloader = DataLoader(test_data, batch_size=validation_size)

def flatten_data(dataloader):
    X = []
    for images, _ in dataloader:
        images = images.view(images.size(0), -1)
        X.append(images)
    return torch.cat(X, dim=0)

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
#Calculating reconstruction error for the pytorch neural network implementation
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

    batched = A.dim() == 3

    if batched:
        inner = torch.bmm(A, f.unsqueeze(-1)).squeeze(-1)
    else:
        inner = A @ f

    intensities = inner ** 2

    a_norms = torch.norm(A, dim=-1)
    f_norm = torch.norm(f, dim=-1, keepdim=True)
    map_sqrt = math.sqrt(A.shape[-1])
    normalized_amp = (map_sqrt * inner.abs()) / (a_norms * f_norm + eps)

    mask1 = (normalized_amp >= alpha_f_lb) & (normalized_amp <= alpha_f_ub)

    residuals = (y - intensities).abs()
    K_t = residuals.mean()

    mask2 = residuals <= alpha_fs * K_t * normalized_amp

    mask = (mask1 & mask2).float()

    valid = intensities > eps
    weights = torch.where(valid, (1.0 - y / (intensities + eps)) * inner,
                          torch.zeros_like(inner))
    weights = weights * mask.float()

    if batched:
        # bmm: (k,n,m) @ (k,m,1) -> (k,n,1) -> (k,n)
        # Never materialises a transposed view larger than (k,n,m).
        return torch.bmm(A.transpose(-2, -1), weights.unsqueeze(-1)).squeeze(-1)
    else:
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

#######################################################################################################################################
#Definition of the PCA model.
#######################################################################################################################################

def compute_pca(X, d):
    """
    Computes the PCA matrix for data matrix X, with embedding dimension k.
    :param X: Data list
    :param k: Embedding dimension
    :param d: Latent dimension
    :return: A tuple with the PCA matrix and the bias
    """
    mu = X.mean(dim=0, keepdim=True)
    X_centered = X - mu

    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

    components = Vt[:d]
    return components, mu

def G_pca(mu, U, z):
    if z.dim() > 1:
        # Handles batched z shape (k, d)
        return mu + (U @ z.unsqueeze(-1)).squeeze(-1)
    # Handles unbatched z shape (d,)
    return mu + U @ z

#######################################################################################################################################
#Reconstruction for the PCA implementation
#######################################################################################################################################

def z_step_pca(mu, U, z, f, lambda_, lr):
    #Sets all previously computed gradients to 0
    if z.grad is not None:
        z.grad.zero_()

    Gz = (mu + (U @ z.unsqueeze(-1)).squeeze(-1))
    #print(Gz.grad_fn)
    loss = (torch.norm(Gz - f.detach(), dim=-1) ** 2).sum()   #detach() may be wrong.
    #Update gradients of z
    loss.backward()
    with torch.no_grad():
        z -= lr * lambda_ * z.grad

    return z.detach().requires_grad_(True)

#Formulation: g(f) + \norm{G(z) - f}_2^2.
def update_formulation1(f, z, A, y, alpha_fs, lambda_, mu, U, stepsize, z_lr):
    grad_f_torch = poisson_wirtinger_grad(A, f, y, alpha_fs) + 2 * lambda_ * (f - G_pca(mu, U, z))

    f = f - stepsize * grad_f_torch
    z = z_step_pca(mu, U, z, f, lambda_, z_lr)

    return f, z

#Formulation: g circ G(z)
def update_formulation2(f, z, A, y, alpha_fs, mu, U, z_lr):
    f = (mu + (U @ z.detach().unsqueeze(-1)).squeeze(-1))
    grad_f = poisson_wirtinger_grad(A, f, y, alpha_fs)
    grad_z = grad_f @ U

    z = z - z_lr * grad_z
    return f, z

def optimize_model_pca(d: int, n: int, m: int, A, y, mu, U):
    #model = G_Model(d,n).to(device)
    #freeze_model(model)
    lambda_ = 0.1 #math.sqrt(n)
    stepsize = 0.2 / (max(lambda_,1) * m)
    z_lr = 0.1 / (max(lambda_, 1) * m)

    f = torch.randn(validation_batch_size, n, requires_grad=False, device=device)
        #torch.from_numpy(new_trunc_spectral_init(A, y, n, m, alpha_fs, True)).to(device)
    z = torch.randn(validation_batch_size, d, requires_grad = True, device=device) #Zoek hier nog een daadwerkelijk goeie initializer voor.

    for _ in range(0, MAX_ITER):
        #We first do one iteration of optimzation for f, with fixed z.
        #For this we use Truncated wirtinger flow.
        f ,z = update_formulation2(f, z, A, y, alpha_fs, mu, U, z_lr)
    z = mu + (U @ z.detach().unsqueeze(-1)).squeeze(-1)
    return z

#######################################################################################################################################
#Definition and training of a deep neural network implementation
#######################################################################################################################################

class AutoEncoder(nn.Module):
    def __init__(self, d, n):
        super(AutoEncoder, self).__init__()
        self.flatten = nn.Flatten()
        step = (n-d) // 4
        hidden1 = n - step
        hidden2 = n - 2 * step
        hidden3 = n - 3 * step

        self.encoder = nn.Sequential(
            nn.Linear(n, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, d),
        )

        self.decoder = nn.Sequential(
            nn.Linear(d, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, n),
            nn.Softplus(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

#Training parameters
learning_rate = 1e-3
epochs = 700

def train_loop(dataloader, model, optimizer, loss_fn):
    for epoch in range(epochs):
        for images, _ in dataloader:
            images = images.view(images.size(0), -1).to(device)

            pred = model(images)
            loss = loss_fn(pred, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
    return model


def train_ae(d: int, n: int):
    model = AutoEncoder(d, n).to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    return train_loop(train_dataloader, model, optimizer, loss_function)

#######################################################################################################################################
#Utility/Caller functions to generate statistics on the reconstruction error
#######################################################################################################################################

def calc_reconstruction_error(inputs) -> (int, int, float):
    n = 784
    d = inputs[0][0]
    components, mu = inputs[0][1]

    U = components.T.to(device=device, dtype=torch.float32)
    mu = mu.squeeze(0).to(device=device, dtype=torch.float32)

    oversampling = inputs[1]
    m = int(oversampling * n)

    images, _ = next(iter(test_dataloader))
    images = images.view(images.size(0), -1)
    ground_truths = images.to(device=device, dtype=torch.float32)

    errors = []

    for gt in ground_truths:
        #print(torch.norm(gt))
        A = generate_measurement_matrix_gpu(n, m, validation_batch_size)
        y = generate_measurement_gpu(A, gt)

        estimators = optimize_model_pca(d, n, m, A, y, mu, U)
        #print(gt)
        #print(estimators)
        del A,y
        errs = (torch_recon_error(estimators, gt.unsqueeze(0)) / torch.norm(gt)).tolist()
        errors.extend(errs)

    average = sum(errors) / len(errors)

    print(str(inputs[0][0]) +"," + str(inputs[1]) + " , Completed " + ", Average: " + str(average))
    return d, oversampling, average


def run_simulation():
     neural_net_dim = [112, 224, 336, 448, 560, 672, 768]
     X_train = flatten_data(train_dataloader).to(device)
     components, mu = compute_pca(X_train, 768)
     models = [(components[:d], mu) for d in neural_net_dim]
     oversampling = [0.5, 1, 1.5, 2, 3, 4, 6]

     "Starting reconstruction"

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
    train_ae(448, 784)
