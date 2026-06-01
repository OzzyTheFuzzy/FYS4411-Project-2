import json
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from training import Training
from model import SE_Model, reconstruct_SE_model
from pathlib import Path
from load_positions import load_pos_general

project_root = Path(__file__).resolve().parent

data_dir = project_root / "positions"
model_dir = project_root / "models"

# Helper functions for loading results, plotting, and reconstructing models from checkpoints

def load_results(model_name):
    with open(f"logs/{model_name}.json", "r") as f:
        results = json.load(f)
    return results


def plot_loss_curves(model_name, config):
    results = load_results(model_name) #loading results from json file
    a=config.a
    loss = results["train_loss"]
    epochs = results["epochs"]
    val_loss = results["val_loss"]
    epochs_val = results["epochs_val"]
    
    name_of_plot = f"{model_name}_loss.pdf"
    
    # Plot loss curve
    plt.plot(epochs, loss, "o-", label="Training", color="blue")
    plt.plot(epochs_val, val_loss, "o-", label="Validation loss", color="red")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.title(f"Training and Validation Loss Curves for N={config.N}, d={config.dim}, beta={config.beta}, a={config.a}")
    plt.legend()
    plt.savefig(f"figures/{name_of_plot}")
    plt.show()

    if a!=0.0:
        val_energy = results["energy_mean_val_history"]
        vmc_energy_history = results["vmc_energy_history"]
        epoch_energy=np.linspace(0, epochs_val[-1], len(vmc_energy_history))
        plt.plot(
            epoch_energy, vmc_energy_history,
            "-", label="VMC Energy",
            color="green", linewidth=2.5, alpha=0.75, zorder=2
        )

        plt.plot(
            epochs_val[1:], val_energy[1:],
            "o--", label="Validation Energy",
            color="red", linewidth=1.8, markersize=4,
            markerfacecolor="none", zorder=3
        )

        plt.xlabel("Epoch")
        plt.xlim(0, epochs_val[-1])
        plt.ylabel("Energy")

        ymax = max(max(vmc_energy_history), max(val_energy)) * 1.05
        ymin = min(min(vmc_energy_history), min(val_energy)) * 0.95
        plt.ylim(ymin, ymax)

        plt.grid(True, alpha=0.3)
        plt.title("VMC and Validation Energy During Training")
        plt.legend()
        plt.show()


def model_reconstructer(model_name, nparticles, dim, a=0.0, beta=None, omega_z=1.0, omega_ho=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = model_dir / model_name
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    structure = reconstruct_SE_model(path)
   

    model = SE_Model(
        dim=dim,
        N=nparticles,
        rho_hidden=structure["rho_hidden"],
        phi_hidden=structure["phi_hidden"],
        eta_hidden=structure["eta_hidden"],
        phi_output=structure["phi_output"],
        eta_output=structure["eta_output"], 
        a=a,
        omega_z=omega_z,
        omega_ho=omega_ho,
        activation_function=nn.GELU(), # need to make sure it is correct activation function
        alpha=0.5,
        beta=beta,
    ).to(device)

    model.load_state_dict(state_dict) #apply trained weights to model
    model.eval()
    
    return model


def plot_energies(PINN_energies, energies,a):
    
    samples = np.linspace(0,len(PINN_energies) ,len(PINN_energies))
    plt.scatter(samples, PINN_energies,s=1,label="PINN", color="green")
    ones=np.ones_like(samples)
    e=energies*ones

    if a==0.0:
        plt.plot(samples, e, label=f"Analytical Energy = {energies:.4f}", color="red")
        plt.title(f"PINN vs analytical energy")
    else:
        plt.plot(samples, e, label=f"RBM Energy = {energies:.4f}", color="red")
        plt.title(f"PINN vs RBM energy")

    plt.xlabel("Sample #")
    plt.ylabel("Energy")
    plt.legend()
    plt.show()


def energy_vmc_and_plot(model_name, N, d, samples, beta, a=0.0, omega_z=1.0, omega_ho=1.0, device="cpu", full=False):
    energies_for_plot = []
    
    if a==0.0:
       
        if beta == 1.0:
            E_ana = N * d / 2
        else:
            E_ana = N * (1 + beta / 2)
    else:
        if N == 2 and d == 3 and a == 1.0 and beta != 1.0:
            E_ana = 5.688  # obtained from RBM, can be adjusted based on the specific system and model initialization
        elif N == 10 and d == 3 and a == 1.0 and beta != 1.0:
            E_ana = 58.48  # obtained from RBM, can be adjusted based on the specific system and model initialization
        else:
            raise ValueError("Analytical energy not known for this combination of parameters")
        
    max_plot_points = 10000
    batch_size      = 10000 # to avoid memory issues when evaluating the energy on a large number of samples
    model_name=f'{model_name}.pth'

    model = model_reconstructer(model_name, N, d, a=a, beta=beta, omega_z=omega_z, omega_ho=omega_ho)

    positions = load_pos_general(a, beta, N, d)

    print(f"Loaded positions with shape: {positions.shape}")
    E_sum = 0.0
    E2_sum = 0.0
    count = 0
    E_K_sum = 0.0
    V_sum = 0.0
    V_coulomb_sum = 0.0
    trainer = Training(model)
    all_energies = []
    if full == False:

        positions_i = torch.tensor(positions[-10_000:],dtype=torch.float32,device=device,requires_grad=True,)
        
        E_L, E_K, V, V_coulomb = trainer.energy_model(positions_i)
        
        E_i   = E_L.detach().cpu().numpy().reshape(-1)
        all_energies.append(E_i)

        E_K_i = E_K.detach().cpu().numpy().reshape(-1)
        V_i   = V.detach().cpu().numpy().reshape(-1)
        V_coulomb_i = V_coulomb.detach().cpu().numpy().reshape(-1)
        
        E_L_mean = E_L.mean().item()
        E_K_mean = E_K.mean().item()
        V_mean = V.mean().item()
        V_coulomb_mean = V_coulomb.mean().item()
       
        E_L_var = E_L.var(unbiased=False).item()

        E_std = np.sqrt(max(E_L_var, 0.0))
        
        print(f"Using last 10,000 positions for quick energy evaluation:")
        print(f"N = {N}, d = {d}, beta = {beta}, a = {a}")
        print(f"PINN E_mean = {E_L_mean:.5f} and E_estimate = {E_ana:.5f}") #estimate is from RBM-VMC unless a=0, then it is analytical energy
        print(f"PINN E_std  = {E_std:.5f}, E_var = {E_L_var:.5f}" )
        print(f"PINN E_K_mean = {E_K_mean:.5f}")
        print(f"PINN V_mean = {V_mean:.5f}")
        print(f"PINN V_coulomb_mean = {V_coulomb_mean:.5f}")
        plot_energies(E_i, E_ana,a=a)


        return E_L_mean, E_std, np.array(all_energies)
        
    # if full = True we evaluate the energy on the full dataset (can be very slow, so we do it in batches and only save a subset of energies for plotting)
    all_energies = []
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)

        print(f"Loading positions: {start} / {samples}", end="\r")

        positions_i = torch.tensor(positions[start:stop],dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        
        E_L, E_K, V, V_coulomb = trainer.energy_model(positions_i)

        E_i = E_L.detach().cpu().numpy().reshape(-1)

        # store ALL energies
        all_energies.append(E_i)
        E_K_i = E_K.detach().cpu().numpy().reshape(-1)
        V_i = V.detach().cpu().numpy().reshape(-1)
        V_coulomb_i = V_coulomb.detach().cpu().numpy().reshape(-1)

        # save subset of VMC energies for plotting
        remaining = max_plot_points - len(energies_for_plot)
        if remaining > 0:
            energies_for_plot.extend(E_i[:remaining])

        E_sum += E_i.sum()
        E2_sum += np.sum(E_i**2)
        count += E_i.size
        E_K_sum += E_K_i.sum()
        V_sum += V_i.sum()
        V_coulomb_sum += V_coulomb_i.sum()

        # delete for more memory
        del positions_i, E_L, E_K, V, V_coulomb

    E_all = np.concatenate(all_energies)
    E_mean = E_sum / count
    E_var = E_all.var(ddof=0)
    E_std = E_all.std(ddof=0)
    E_K_mean = E_K_sum / count
    V_mean = V_sum / count
    V_coulomb_mean = V_coulomb_sum / count
    E_std = np.sqrt(E_var)

    print("\npositions obtained")
    print(f"N = {N}, d = {d}, beta = {beta}, a = {a}")
    print(f"PINN E_mean = {E_mean:.8f} and E_estimate = {E_ana:.8f}")
    print(f"PINN E_std  = {E_std}, E_var = {E_var}")
    print(f"PINN E_K_mean = {E_K_mean:.8f}")
    print(f"PINN V_mean = {V_mean:.8f}")
    print(f"PINN V_coulomb_mean = {V_coulomb_mean:.8f}")
    plot_energies(energies_for_plot, E_ana,a=a)
   
    return E_mean, E_std, np.concatenate(all_energies)



