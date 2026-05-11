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

def load_pos_memmap(n_samples, N, d, beta=None, a=0.0):
    """
    Load memmap WITHOUT loading into RAM.
    """

    beta_str = "None" if beta is None else str(beta)
    a_str = str(a)

    filename = f"r_all_N{N}_d{d}_beta{beta_str}_a{a_str}.dat"
    path = data_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")

    r_all = np.memmap(
        path,
        dtype="float64",
        mode="r",
        shape=(n_samples, N, d),
    )

    return r_all


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


def energy_vmc_and_plot(model_name, N, d, samples, beta, a=0.0, omega_z=1.0, omega_ho=1.0, device="cpu", full=False):
    energies_for_plot = []

    if beta == 1.0:
        E_ana= N * d / 2
    else:
        E_ana = N * (1 + beta / 2)

    max_plot_points = 10000
    batch_size = 10000
    model_name=f'{model_name}.pth'

    model = model_reconstructer(model_name, N, d, a=a, beta=beta, omega_z=omega_z, omega_ho=omega_ho)

    if beta ==1.0:
        beta=None
        positions= load_pos_memmap(
            n_samples=samples,
            N=N,
            d=d,
            beta=beta,
            a=a,
        )

    else:
        positions= load_pos_memmap(
            n_samples=samples,
            N=N,
            d=d,
            beta=beta,
            a=a,
        )

    E_sum = 0.0
    E2_sum = 0.0
    count = 0
    E_K_sum = 0.0
    V_sum = 0.0
    V_coulomb_sum = 0.0
    trainer = Training(model)

    if full == False:

        positions_i = torch.tensor(positions[-10_000:],dtype=torch.float32,device=device,requires_grad=True,)

        E_L, E_K, V, V_coulomb = trainer.energy_model(positions_i)

        E_i   = E_L.detach().cpu().numpy().reshape(-1)
        E_K_i = E_K.detach().cpu().numpy().reshape(-1)
        V_i   = V.detach().cpu().numpy().reshape(-1)
        V_coulomb_i = V_coulomb.detach().cpu().numpy().reshape(-1)

        E_L_mean = E_L.mean().item()
        E_K_mean = E_K.mean().item()
        V_mean = V.mean().item()
        V_coulomb_mean = V_coulomb.mean().item()
        E_L_var = E_L.var().item()

        E_std = np.sqrt(max(E_L_var, 0.0))
        print(f"Using last 10,000 positions for quick energy evaluation:")
        print(f"N = {N}, d = {d}, beta = {beta}, a = {a}")
        print(f"PINN E_mean = {E_L_mean:.8f} and E_ana = {E_ana:.8f}")
        print(f"PINN E_std  = {E_std:.8f}")
        print(f"PINN E_K_mean = {E_K_mean:.8f}")
        print(f"PINN V_mean = {V_mean:.8f}")
        print(f"PINN V_coulomb_mean = {V_coulomb_mean:.8f}")
        plot_energies(E_i, E_ana)

        return E_L_mean, E_std
        
    # if full = True we evaluate the energy on the full dataset (can be very slow, so we do it in batches and only save a subset of energies for plotting)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)

        print(f"Loading positions: {start} / {samples}", end="\r")

        positions_i = torch.tensor(positions[start:stop],dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        
        E_L, E_K, V, V_coulomb = trainer.energy_model(positions_i)

        E_i = E_L.detach().cpu().numpy().reshape(-1)
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

    E_mean = E_sum / count
    E_var = E2_sum / count - E_mean**2
    E_K_mean = E_K_sum / count
    V_mean = V_sum / count
    V_coulomb_mean = V_coulomb_sum / count
    E_std = np.sqrt(E_var)

    print("\npositions obtained")
    print(f"N = {N}, d = {d}, beta = {beta}, a = {a}")
    print(f"PINN E_mean = {E_mean:.8f} and E_ana = {E_ana:.8f}")
    print(f"PINN E_std  = {E_std:.8f}")
    print(f"PINN E_K_mean = {E_K_mean:.8f}")
    print(f"PINN V_mean = {V_mean:.8f}")
    print(f"PINN V_coulomb_mean = {V_coulomb_mean:.8f}")
    plot_energies(energies_for_plot, E_ana)

    return E_mean, E_std



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


