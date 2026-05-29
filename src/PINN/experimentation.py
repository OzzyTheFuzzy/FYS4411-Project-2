import json
import torch 
import torch.nn as nn
import numpy as np
from training import Training
from initialize_data import InitializeData
from model import SE_Model
from PINN_vs_analytical import *
from blocking import blocking_error, plot_blocking

# Configuration
width = 1.3     # Width of the Gaussian distribution for sampling collocation points
a     = 1.0     # a=1.0  for strength of the Coulomb interactions   
N     = 10        # Number of particles (dimensions)
dim   = 3        # Dimensionality of the particles
omega_ho = 1.0        # Frequency of the harmonic trap in the x and y directions
beta     = 2.82843  #set beta=1.0 for isotropic case
omega_z  = beta       # Frequency of the harmonic trap in the z-direction. Set equal to beta for antisotropic case, and to 1 for isotropic cas
beta_jastrow = 0.5 # Wavefunction parameter when interactions are included 
alpha   = 0.497000 # Wavefunction parameter for the single-particle part of the wavefunction

#  Training parameters
training_points = 4000
seed            = 17
epochs      = 1000
batch_size  = 500
num_batches = training_points // batch_size
val_points  = 2000
val_width   = width # width of the Gaussian distribution for sampling validation collocation points
val_seed    = 42 # random seed for sampling validation collocation points
gamma       = 0.995 # learning rate decay factor for scheduler, set to a value close to 1 for smoother convergence
lr          = 2.5e-3 # learning rate for optimizer. Will be tuned during training by scheduler for smoother convergence
lr_E        = 1e-5 # learning rate for energy parameter, set lower than lr for smoother convergence towards true GS energy
lr_alpha    = 1e-6 # learning rate for alpha parameter, set lower than
trainable_alpha = False # whether to train the energy parameter alpha or keep it fixed during training
trainable_energy = False # whether to train the energy parameter or keep it fixed during training
coulomb_init    = False # if we do not have hard coded energy results for the given config, use coulomb initialization
initialize_gaussian = False
trainable_beta_jastrow = False
lr_beta_jastrow = 1e-5

model_name      = f"{N}N_d{dim}_beta{beta}_a{a}_width{width}_IG{initialize_gaussian}" # name for saving model and logs
#model_name="2N_d1_beta1.0_a0.0_width0.95_IGTrue"
def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create instance of data initialization 
    print(f"Initializing data for N={N}, dim={dim}, a={a} and beta={beta}")
    data_initializer = InitializeData(N, dim, device=device, dtype=torch.float32, interacting_strength=a, initialize_gaussian=initialize_gaussian, omega_z=omega_z)
    print("Data initialization complete.")
    # create data for training
    
    model_config = {
        "dim": dim,
        "N": N,
        "rho_hidden": [32, 32], 
        "phi_hidden": [4, 4], #
        "eta_hidden": [4, 4], #
        "phi_output": 4,
        "eta_output": 4,
        "activation_function": nn.GELU(),
        "alpha": alpha,
        "beta": beta,
        "trainable_alpha": trainable_alpha,
        "a": a,
        "omega_z": omega_z,
        "omega_ho": omega_ho,
        "trainable_energy": trainable_energy,
        "beta_jastrow": beta_jastrow,
        "trainable_beta_jastrow": trainable_beta_jastrow,
    }
    # Model initialization, with optimizer and scheduler
    model = SE_Model(**model_config)
    print("Model initialization complete.")

    # Create an instance of the training class
    trainer = Training(model, val_points, val_width, val_seed)
    print("Training class initialization complete.")
    # slowing down training for alpha and energy parameter for smoother convergence towards true GS
    param_groups = [
    {
        "params": [
            p for name, p in trainer.named_parameters()
            if name not in [
                "energy",
                "model.alpha",
                "model.beta_jastrow",
            ]
        ],
        "lr": lr,
    }
    ]

    if trainable_energy:
        param_groups.append({"params": [trainer.energy], "lr": lr_E})

    if trainable_alpha:
        param_groups.append({"params": [trainer.model.alpha], "lr": lr_alpha})

    if trainable_beta_jastrow:
        param_groups.append({"params": [trainer.model.beta_jastrow], "lr": lr_beta_jastrow})

    optimizer = torch.optim.Adam(param_groups)
    # Set up optimizer and learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # train the model and get training results
    loss, epoch_array, val_loss, epochs_val, best_model_state = trainer.training_cycle_SE(
            N_epochs=epochs,
            data_initializer=data_initializer,
            training_points=training_points,
            width=width,
            seed=seed,
            optimizer=optimizer,
            scheduler=scheduler,
            num_batches=num_batches,
            use_scheduler=True,
            coulomb_init=coulomb_init
        )
    print("Training complete.")
    # Save model weights
    checkpoint = {"model_state_dict": best_model_state}
    torch.save(checkpoint, f"models/{model_name}.pth")

    # Save training results + metadata to JSON
    json_config = model_config.copy()
    json_config["activation_function"] = "GELU" #json cannot serialize the activation function, so we save its name instead
    
    results = {
        "model_name": model_name,
        "model_config": json_config,
        "training_points": int(training_points),
        "batch_size": int(batch_size),
        "width": float(width),
        "val_seed": int(val_seed),
        "val_points": int(val_points),
        "initialize_gaussian": bool(initialize_gaussian),
        "seed": int(seed),
        "val_seed": int(val_seed),
        "energy_parameter_final": float(trainer.energy.detach().cpu().item()),
        "energy_parameter_history": list(np.array(trainer.energy_model_array).tolist()),
        "vmc_energy_history": list(trainer.vmc_energy_list),
        "alpha_initial": float(model_config["alpha"]),
        "alpha_final": float(model.alpha.detach().cpu().item()),
        "alpha_history": list(np.array(trainer.alpha_array).tolist()),
        "beta": float(model.beta),
        "energy_std_val_final": float(np.sqrt(trainer.energy_val_var[-1])),

        "learning_rate": float(lr),
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "energy_mean_val_final": float(trainer.energy_val_mean[-1]),
        "energy_var_val_final": float(trainer.energy_val_var[-1]),
        "energy_mean_val_history": list(np.array(trainer.energy_val_mean).tolist()),
        "energy_std_val_history": list(np.array(trainer.energy_val_std_array).tolist()),
        "energy_var_val_history": list(np.array(trainer.energy_val_var).tolist()),

        "epochs": list(np.array(epoch_array).tolist()),
        "train_loss": list(np.array(loss).tolist()),
        "val_loss": list(np.array(val_loss).tolist()),
        "epochs_val": list(np.array(epochs_val).tolist()),
    }
    
    results["final_learning_rate"] = optimizer.param_groups[0]["lr"]

    with open(f"logs/{model_name}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved model to models/{model_name}.pth")
    print(f"Saved logs to logs/{model_name}.json")

train_and_evaluate() # uncomment for training 
plot_loss_curves(model_name) # for plotting the loss during training

# vmc samples from .dat file
if a==0.0:
    samples = 800000

else:
    samples = 524288

full = False # set true for eval over many samples
E_mean, E_std, E_L= energy_vmc_and_plot(model_name, N=N, d=dim, samples=samples, beta=beta, a=a, omega_z=omega_z, omega_ho=omega_ho, full=full) # for evaluating the energy of the trained model
block_variance, block_error, B_list, n_list = blocking_error(E_L.flatten()) # for performing blocking analysis on the energies obtained from the trained model

plot_blocking(B_list, block_error, beta=beta, name_of_model=model_name, N=N, a=a) # for plotting the blocking analysis results
