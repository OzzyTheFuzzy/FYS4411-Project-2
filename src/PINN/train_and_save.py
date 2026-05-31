import torch
from training import Training
from initialize_data import InitializeData
import numpy as np
import json
from model import SE_Model
import torch.nn as nn

def train_and_save(config, model_name, model_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create instance of data initialization 
    print(f"Initializing data for N={config.N}, dim={config.dim}, a={config.a} and beta={config.beta}")
    data_initializer = InitializeData(config.N, config.dim, device=device, dtype=torch.float32, interacting_strength=config.a, initialize_gaussian=config.initialize_gaussian, omega_z=config.omega_z)
    print("Data initialization complete.")
    # create data for training
    
    
   
    # Model initialization, with optimizer and scheduler
    model = SE_Model(**model_config)
    print("Model initialization complete.")

    # Create an instance of the training class
    
    trainer = Training(model, config.val_points, config.val_width, config.val_seed)
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
        "lr": config.lr,
    }
    ]

    if config.trainable_energy:
        param_groups.append({"params": [trainer.energy], "lr": config.lr_E})

    if config.trainable_alpha:
        param_groups.append({"params": [trainer.model.alpha], "lr": config.lr_alpha})

    if config.trainable_beta_jastrow:
        param_groups.append({"params": [trainer.model.beta_jastrow], "lr": config.lr_beta_jastrow})

    optimizer = torch.optim.Adam(param_groups)
    # Set up optimizer and learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    # train the model and get training results
    print(f"Starting training for model {model_name} ")
    loss, epoch_array, val_loss, epochs_val, best_model_state = trainer.training_cycle_SE(
            N_epochs=config.epochs,
            data_initializer=data_initializer,
            training_points=config.training_points,
            width=config.width,
            seed=config.seed,
            optimizer=optimizer,
            scheduler=scheduler,
            num_batches=config.num_batches,
            use_scheduler=True,
            coulomb_init=config.coulomb_init
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
        "training_points": int(config.training_points),
        "batch_size": int(config.batch_size),
        "width": float(config.width),
        "val_seed": int(config.val_seed),
        "val_points": int(config.val_points),
        "initialize_gaussian": bool(config.initialize_gaussian),
        "seed": int(config.seed),
        "val_seed": int(config.val_seed),
        "energy_parameter_final": float(trainer.energy.detach().cpu().item()),
        "energy_parameter_history": list(np.array(trainer.energy_model_array).tolist()),
        "vmc_energy_history": list(trainer.vmc_energy_list),
        "alpha_initial": float(model_config["alpha"]),
        "alpha_final": float(model.alpha.detach().cpu().item()),
        "alpha_history": list(np.array(trainer.alpha_array).tolist()),
        "beta": float(model.beta),
        "energy_std_val_final": float(np.sqrt(trainer.energy_val_var[-1])),

        "learning_rate": float(config.lr),
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