import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

from training import Training
from model import SE_Model, reconstruct_SE_model


def evaluate_energy(model_name, positions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = f"models/{model_name}.pth"
    checkpoint = torch.load(path, map_location=device)

    state_dict = checkpoint["model_state_dict"]

    info = reconstruct_SE_model(state_dict)

    positions = positions.to(device)
    nparticles = positions.shape[1]
    dim = positions.shape[2]

    model = SE_Model(
        dim=dim,
        N=nparticles,
        rho_hidden=info["rho_hidden"],
        phi_hidden=info["phi_hidden"],
        eta_hidden=info["eta_hidden"],
        phi_output=info["phi_output"],
        eta_output=info["eta_output"], 
        activation_function=nn.GELU(), # need to make sure it is correct activation function
        alpha=0.5,
        beta=1.0
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    trainer = Training(model)

    energy = trainer.energy_model(positions)

    return energy

