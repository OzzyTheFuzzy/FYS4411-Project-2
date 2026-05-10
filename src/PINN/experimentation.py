import json
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from training import Training
from initialize_data import InitializeData
from model import SE_Model, reconstruct_SE_model
from PINN_vs_analytical import *

model_name  = "10N_beta2.82843_lr3_20k_width0.75" # name for saving model and logs
samples=1000000
samples=int(1000000-samples//10*2.0)# to check VMC energy (samples=amount of positions)

def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration
    width = 0.65      # Width of the Gaussian distribution for sampling collocation points
    a     = 0.0  #0.0043  for interactions   Hard-core radius (set to 0 for no interactions)
    N     = 10        # Number of particles (dimensions)
    dim   = 3         # Dimensionality of the particles
    omega_z = 2.82843      # Frequency of the harmonic trap in the z-direction
    omega_ho = 1.0    # Frequency of the harmonic trap in the x and y directions
    
    
    #  Training parameters
    training_points = 20000
    seed            = 17
    epochs      = 200
    batch_size  = 1000
    num_batches = training_points // batch_size
    val_points  = 7500
    val_width   = 0.75  # width of the Gaussian distribution for sampling validation collocation points
    val_seed    = 42 # random seed for sampling validation collocation points
    lr          = 1e-3 # learning rate for optimizer. Will be tuned during training by scheduler for smoother convergence
    
    # Create instance of data initialization 
    print(f"Initializing data for N={N}, dim={dim}, a={a}")
    data_initializer = InitializeData(N, dim, device=device, dtype=torch.float32, hard_core_radius=a, omega_z=omega_z)
    print("Data initialization complete.")
    # create data for training
    

    model_config = {
        "dim": dim,
        "N": N,
        "rho_hidden": [32, 32, 32, 32], 
        "phi_hidden": [8], #
        "eta_hidden": [8],
        "phi_output": 4,
        "eta_output": 4,
        "activation_function": nn.GELU(),
        "alpha": 0.5,
        "beta": 2.82843,
        "trainable_alpha": False,
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
    
    # Set up optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(trainer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
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



def energy_vmc_and_plot(model_name, N, d,samples, beta, a=0.0, omega_z=1.0, omega_ho=1.0, device="cpu"):
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
    trainer = Training(model)

    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)

        print(f"Loading positions: {start} / {samples}", end="\r")

        positions_i = torch.tensor(positions[start:stop],dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        
        E_L, E_K, V= trainer.energy_model(positions_i)

        E_i = E_L.detach().cpu().numpy().reshape(-1)
        E_K_i = E_K.detach().cpu().numpy().reshape(-1)
        V_i = V.detach().cpu().numpy().reshape(-1)

        # save subset of VMC energies for plotting
        remaining = max_plot_points - len(energies_for_plot)
        if remaining > 0:
            energies_for_plot.extend(E_i[:remaining])

        E_sum += E_i.sum()
        E2_sum += np.sum(E_i**2)
        count += E_i.size
        E_K_sum += E_K_i.sum()
        V_sum += V_i.sum()
        # delete so we dont run out of memory
        del positions_i, E_L, E_K, V

    E_mean = E_sum / count
    E_var = E2_sum / count - E_mean**2
    E_K_mean = E_K_sum / count
    V_mean = V_sum / count
    E_std = np.sqrt(E_var)

    print("\npositions obtained")
    print(f"N = {N}, d = {d}, beta = {beta}, a = {a}")
    print(f"PINN E_mean = {E_mean:.8f} and E_ana = {E_ana:.8f}")
    print(f"PINN E_std  = {E_std:.8f}")
    print(f"PINN E_K_mean = {E_K_mean:.8f}")
    print(f"PINN V_mean = {V_mean:.8f}")
    plot_energies(energies_for_plot, E_ana)

    return E_mean, E_std

#train_and_evaluate() # uncomment for training 
plot_loss_curves(model_name) # for plotting the loss during training
beta=2.82843
E_mean, E_std= energy_vmc_and_plot(model_name, N=10, d=3, samples=samples, beta=beta, a=0.0, omega_z=beta, omega_ho=1.0) # for evaluating the energy of the trained model


