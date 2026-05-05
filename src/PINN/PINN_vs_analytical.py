import json

import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from training import Training
from model import SE_Model, reconstruct_SE_model
from pathlib import Path
from scipy.integrate import trapezoid


project_root = Path(__file__).resolve().parent

data_dir = project_root / "positions_energy_data"
model_dir = project_root / "models"
#filename= 'r_all_E_N2_d1_betaNone_a0.0' #without .npz

#model_name = "2N_1D_GELU_32323232_12output.pth" # name of model with .pth


def load_results(model_name):
    with open(f"logs/{model_name}.json", "r") as f:
        results = json.load(f)
    return results

def plot_loss_curves(model_name):
    results = load_results(model_name)

    loss = results["train_loss"]
    epochs = results["epochs"]
    val_loss = results["val_loss"]
    epochs_val = results["epochs_val"]

    # Plot loss curve
    plt.plot(epochs, loss, "o-", label="Training", color="blue")
    plt.plot(epochs_val, val_loss, "o-", label="Validation loss", color="red")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.show()

def find_npz_file(data_dir, N, d, beta=None, a=0.0):
    """
    Finds file with pattern:
    r_all_E_N{N}_d{d}_beta{beta}_a{a}.npz
    """

    data_dir = Path(data_dir)

    beta_str = "None" if beta is None else str(beta)
    a_str = str(a)

    filename = f"r_all_E_N{N}_d{d}_beta{beta_str}_a{a_str}.npz"
    path = data_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")

    return path

def load_positions_and_energy_from_params(N, d, beta=None, a=0.0, device="cpu"):
    path = find_npz_file(data_dir, N=N, d=d, beta=beta, a=a)

    data = np.load(path)

    r_all = data["r_all"]
    E_ana = data["E"]

    positions = torch.tensor(r_all, dtype=torch.float32, device=device)
    energies = torch.tensor(E_ana, dtype=torch.float32, device=device).reshape(-1, 1)

    n_samples, N_loaded, dim_loaded = r_all.shape

    return positions, energies.detach().cpu().numpy(), n_samples, N_loaded, dim_loaded

def evaluate_energy(model_name, positions, a=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = model_dir / model_name
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    structure = reconstruct_SE_model(path)
   
    positions = positions.to(device)
    nparticles = positions.shape[1]
    dim = positions.shape[2]
    a = 0.0
    model = SE_Model(
        dim=dim,
        N=nparticles,
        a=a,
        rho_hidden=structure["rho_hidden"],
        phi_hidden=structure["phi_hidden"],
        eta_hidden=structure["eta_hidden"],
        phi_output=structure["phi_output"],
        eta_output=structure["eta_output"], 
        activation_function=nn.GELU(), # need to make sure it is correct activation function
        alpha=0.5,
        beta=1.0
    ).to(device)

    model.load_state_dict(state_dict) #apply trained weights to model
    model.eval()

    trainer = Training(model) #initialize trainer to access energy_model method
    positions = positions.to(device)
    positions = positions.detach().clone().requires_grad_(True)
    energy, E_K, V = trainer.energy_model(positions) #retrieve energy for the given positions

    with torch.no_grad():
        u = model(positions) #calculate the log(pos) for the wavefunction for the given positions

    return energy.detach().cpu().numpy(), u.detach().cpu().numpy() #return energy and log(pos) as numpy arrays for plotting


def load_positions_and_energy(filename, device="cpu", index=None):
    """
    if you have the filename and only want pos and energy
    """
    data = np.load(data_dir / f"{filename}.npz")
    
    r_all = data["r_all"]
    n_samples, N, dim = r_all.shape 

    E_ana = data["E"]

    positions = torch.tensor(r_all, dtype=torch.float32, device=device)
    energies = torch.tensor(E_ana, dtype=torch.float32, device=device).reshape(-1, 1)

    if index is not None:
        positions = positions[index:index+1]
        energies = energies[index:index+1]

    return positions, energies.detach().cpu().numpy(), n_samples, N, dim # retrieve positions with torch and and return energies as numpy for plotting


def plot_energies(PINN_energies, energies):
    
    samples = np.linspace(0,len(PINN_energies) ,len(PINN_energies))
    plt.scatter(samples, PINN_energies,s=1,label="PINN", color="green")
    plt.plot(energies, label="Analytical", color="red")
    plt.title("PINN vs analytical energy")
    plt.xlabel("Sample #")
    plt.ylabel("Energy")
    plt.legend()
    plt.show()


def test(model_name="2N_1D_GELU_323232_8output"):

    N = 2
    d = 1
    beta = None
    a = 0.0
    positions, energies, n_samples, N, dim = load_positions_and_energy_from_params(
        N=N,
        d=d,
        beta=beta,
        a=a
    )

    PINN_energies, PINN_log_wf = evaluate_energy(model_name, positions, a=a)
    
    E_mean = PINN_energies.mean()
    E_std = PINN_energies.std()

    print(f"N = {N}, d = {dim}, beta = {beta}, a = {a}")
    print(f"PINN E_mean = {E_mean:.8f}")
    print(f"PINN E_std  = {E_std:.8f}")


def plot_psi_sqare(positions, PINN_log_wf_array):
    dim=1
    N=2
    x1 = positions[:,0,0].cpu().numpy() #xpositions of first particle
    psi2 = np.exp(2 * PINN_log_wf_array)

    # bin x1
    bins = 100
    hist, edges = np.histogram(x1, bins=bins, weights=psi2, density=True)
    centers = 0.5*(edges[:-1] + edges[1:])

    x1 = positions[:, 0, 0].cpu().numpy()
    x2 = positions[:, 1, 0].cpu().numpy()

    r = np.sqrt(x1**2 + x2**2)
    bins = 100
    hist, edges = np.histogram(r, bins=bins, density=True)

    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.plot(centers, hist)
    plt.xlabel("r")
    plt.ylabel("P(r)")
    plt.title("Radial probability density")
    plt.grid(True)
    plt.show()

    if dim==1 and N==1:
        pos_array = positions.detach().cpu().numpy().reshape(-1)
        
        wf_square = np.exp(2 * PINN_log_wf_array)

        # Normalize PINN |psi|^2
        norm = trapezoid(wf_square, pos_array)
        wf_norm = wf_square / norm

        # Analytical |psi|^2
        x = np.linspace(-3, 3, 500)
        psi_sq_true = (1 / np.sqrt(np.pi)) * np.exp(-x**2)

        # Plot unnormalized PINN
        plt.scatter(pos_array, wf_square, label="PINN unnormalized", color="red")
        plt.plot(x, psi_sq_true, label="Analytical", color="green")
        plt.xlabel("x")
        plt.ylabel(r"$|\psi(x)|^2$")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot normalized comparison
        plt.scatter(pos_array, wf_norm, s=5, label="PINN normalized",color="red")
        plt.plot(x, psi_sq_true, label="Analytical", color="green")
        plt.xlabel("x")
        plt.ylabel(r"$|\psi(x)|^2$")
        plt.legend()
        plt.grid(True)
        plt.show()

    return 0