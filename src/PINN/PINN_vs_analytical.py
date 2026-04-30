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
filename= 'r_all_E_N2_d1_betaNone_a0.0' #without .npz

model_name = "2N_1D_GELU_323232_test.pth" # name of model with .pth

def evaluate_energy(model_name, positions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = model_dir / model_name
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    structure = reconstruct_SE_model(path)
    print(positions.shape)
    positions = positions.to(device)
    nparticles = positions.shape[1]
    dim = positions.shape[2]

    model = SE_Model(
        dim=dim,
        N=nparticles,
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

positions, energies, n_samples, N, dim = load_positions_and_energy(filename)

PINN_energies, PINN_log_wf = evaluate_energy(model_name, positions)


PINN_log_wf_array = PINN_log_wf.reshape(-1)
samples = np.linspace(0,len(PINN_energies) ,len(PINN_energies))
plt.scatter(samples, PINN_energies,s=2,label="PINN")
plt.plot(energies, label="Analytical")
plt.xlabel("Sample")
plt.ylabel("Energy")
plt.legend()
plt.show()


if dim==1 and N==1:
    pos_array = positions.detach().cpu().numpy().reshape(-1)
    print(pos_array)
    wf_square = np.exp(2 * PINN_log_wf_array)

    # Normalize PINN |psi|^2
    norm = trapezoid(wf_square, pos_array)
    wf_norm = wf_sorted / norm

    # Analytical |psi|^2
    x = np.linspace(-3, 3, 500)
    psi_sq_true = (1 / np.sqrt(np.pi)) * np.exp(-x**2)

    # Plot unnormalized PINN
    plt.scatter(pos_array, wf_sorted, label="PINN unnormalized", color="red")
    plt.plot(x, psi_sq_true, label="Analytical", color="green")
    plt.xlabel("x")
    plt.ylabel(r"$|\psi(x)|^2$")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot normalized comparison
    plt.scatter(pos_array, wf_sorted_norm, s=5, label="PINN normalized",color="red")
    plt.plot(x, psi_sq_true, label="Analytical", color="green")
    plt.xlabel("x")
    plt.ylabel(r"$|\psi(x)|^2$")
    plt.legend()
    plt.grid(True)
    plt.show()
