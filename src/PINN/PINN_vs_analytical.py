import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from training import Training
from model import SE_Model, reconstruct_SE_model
from pathlib import Path

project_root = Path(__file__).resolve().parent

data_dir = project_root / "positions_energy_data"
model_dir = project_root / "models"
filename= 'r_all_E_N1_d1_betaNone_a0.0' #without .npz
model_name = "1N_1D_GELU_323232_test_.pth" # name of model with .pth
def evaluate_energy(model_name, positions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = model_dir / model_name
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    structure = reconstruct_SE_model(path)

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

    model.load_state_dict(state_dict)
    model.eval()

    trainer = Training(model)

    energy = trainer.energy_model(positions)

    return energy


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

    return positions, energies

positions, energies = load_positions_and_energy(filename)

PINN_energies = evaluate_energy(model_name, positions)
print(PINN_energies)


plt.plot(PINN_energies.detach().cpu().numpy(), label="PINN")
plt.plot(energies.detach().cpu().numpy(), label="Analytical")
plt.xlabel("Sample")
plt.ylabel("Energy")
plt.legend()
plt.show()