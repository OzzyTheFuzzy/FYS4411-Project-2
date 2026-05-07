import json
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from training import Training
from initialize_data import InitializeData
from model import SE_Model, reconstruct_SE_model
from PINN_vs_analytical import *

model_name  = "10N_3D_t100k_1kepoch" # name for saving model and logs

def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration
    width = 1.0      # Width of the Gaussian distribution for sampling collocation points
    a     = 0.0  #0.0043  for interactions   Hard-core radius (set to 0 for no interactions)
    N     = 10        # Number of particles (dimensions)
    dim   = 3         # Dimensionality of the particles
    omega_z = 1.0      # Frequency of the harmonic trap in the z-direction
    omega_ho = 1.0    # Frequency of the harmonic trap in the x and y directions

    
    #  Training parameters
    training_points = 100000
    epochs      = 1000
    batch_size  = 1000
    num_batches = training_points // batch_size
    val_points  = 10000
    val_width   = 1.0  # width of the Gaussian distribution for sampling validation collocation points
    val_seed    = 42 # random seed for sampling validation collocation points
    lr          = 1e-5 # learning rate for optimizer. Will be tuned during training by scheduler for smoother convergence
    
    # Create instance of data initialization 
    print(f"Initializing data for N={N}, dim={dim}, a={a}")
    data_initializer = InitializeData(N, dim, device=device, dtype=torch.float32, hard_core_radius=a)
    print("Data initialization complete.")
    # create data for training
    
    positions = data_initializer.initialize_pde(training_points, width, seed=17)
    print(" pde Data initialization complete.")
    model_config = {
        "dim": dim,
        "N": N,
        "rho_hidden": [32, 32, 32], 
        "phi_hidden": [8], #
        "eta_hidden": [8],
        "phi_output": 4,
        "eta_output": 4,
        "activation_function": nn.GELU(),
        "alpha": 0.5,
        "beta": 1.0,
        "trainable_alpha": True,
        "a": a,
        "omega_z": omega_z,
        "omega_ho": omega_ho
    }
    # Model initialization, with optimizer and scheduler
    model = SE_Model(**model_config)
    print("Model initialization complete.")
    # Create an instance of the training class
    trainer = Training(model, val_points, val_width, val_seed)
    print("Training class initialization complete.")
    optimizer = torch.optim.Adam(trainer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # train the model and get training results
    loss, epoch_array, val_loss, epochs_val, best_model_state = trainer.training_cycle_SE(
            
            N_epochs=epochs,
            positions=positions,
            optimizer=optimizer,
            scheduler=scheduler,
            num_batches=num_batches,
            use_scheduler=True
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

        "energy_parameter_final": float(trainer.energy.detach().cpu().item()),
        "energy_parameter_history": list(np.array(trainer.energy_model_array).tolist()),

        "alpha_initial": float(model_config["alpha"]),
        "alpha_final": float(model.alpha.detach().cpu().item()),
        "alpha_history": list(np.array(trainer.alpha_array).tolist()),
        "a": float(a),
        "beta": float(model.beta.detach().cpu().item()),
        "energy_std_val_final": float(np.sqrt(trainer.energy_val_var[-1])),

        "learning_rate": float(lr),
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),

       
        "energy_mean_val_final": float(trainer.energy_val_mean[-1]),
        "energy_mean_val_history": list(np.array(trainer.energy_val_mean).tolist()),

        "energy_var_val_final": float(trainer.energy_val_var[-1]),
        "energy_var_val_history": list(np.array(trainer.energy_val_var).tolist()),


        "training_points": int(training_points),

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



def check_energy(N, dim, beta=None, a=0.0):

    positions, energies, n_samples, N_loaded, dim_loaded= load_positions_and_energy_from_params(
        N=N,
        d=dim,
        beta=beta,
        a=a
    )

    PINN_energies, PINN_log_wf = evaluate_energy(f'{model_name}.pth', positions, a=a)
    
    E_mean = PINN_energies.mean()
    E_std = PINN_energies.std()

    print(f"N = {N}, d = {dim_loaded}, beta = {beta}, a = {a}")
    print(f"PINN E_mean = {E_mean:.8f}")
    print(f"PINN E_std  = {E_std:.8f}")

    return positions, energies, PINN_energies, PINN_log_wf


train_and_evaluate() # uncomment for training 
plot_loss_curves(model_name) # for plotting the loss during training
positions, energies, PINN_energies, PINN_log_wf = check_energy(N=1, dim=3, beta=None, a=0.0) # for evaluating the energy of the trained model
plot_energies(PINN_energies, energies) # for plotting the energy of the trained model


