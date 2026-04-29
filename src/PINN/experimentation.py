import json
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

from training import Training
from initialize_data import InitializeData
from model import SE_Model, reconstruct_SE_model

model_name  = "1N_1D_GELU_323232_test2_log" # name for saving model and logs

def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration
    width = 1.0      # Width of the Gaussian distribution for sampling collocation points
    a     = 0.0  #0.0043  for interactions   Hard-core radius (set to 0 for no interactions)
    N     = 1         # Number of particles (dimensions)
    dim   = 1         # Dimensionality of the particles

    # Number of training points 
    training_points = 75000

    #  Training parameters
    epochs      = 600
    num_batches = 50
    val_points  = 10000
    val_width   = 1.0 
    val_seed    = 42
    lr          = 1e-4
    

    # Create instance of data initialization 
    data_initializer = InitializeData(N, dim, device=device, dtype=torch.float32, hard_core_radius=a)

    # create data for training
    positions = data_initializer.initialize_pde(training_points, width, seed=17)
    model_config = {
        "dim": dim,
        "N": N,
        "rho_hidden": [32, 32, 32], 
        "phi_hidden": [32, 32], #
        "eta_hidden": [32, 32],
        "phi_output": 10,
        "eta_output": 10,
        "activation_function": nn.GELU(),
        "alpha": 0.5,
        "beta": 1.0,
        "trainable_alpha": True,
    }
    # Model initialization, with optimizer and scheduler
    model = SE_Model(**model_config)


    # Create an instance of the training class
    trainer = Training(model, val_points, val_width, val_seed)
        
    optimizer = torch.optim.Adam(trainer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # train the model and get training results
    loss, epochs, val_loss, epochs_val, best_model_state = trainer.training_cycle_SE(
            
            N_epochs=epochs,
            positions=positions,
            optimizer=optimizer,
            scheduler=scheduler,
            num_batches=num_batches,
            use_scheduler=True
        )
    
    # Save model weights
    checkpoint = {"model_state_dict": best_model_state}
    torch.save(checkpoint, f"models/{model_name}.pth")

    # Save training results + metadata to JSON
    json_config = model_config.copy()
    json_config["activation_function"] = "GELU" #json cannot serialize the activation function, so we save its name instead
    
    results = {
        "model_name": model_name,
        "model_config": json_config,

        "energy_parameter_final": trainer.energy.detach().cpu().item(),
        "energy_parameter_history": trainer.energy_model_array,

        "alpha_initial": model_config["alpha"],
        "alpha_final": model.alpha.detach().cpu().item(),
        "alpha_history": trainer.alpha_array,

        "beta": model.beta.detach().cpu().item(),

        "energy_mean_val_final": trainer.energy_val_mean[-1],
        "energy_mean_val_history": trainer.energy_val_mean,

        "energy_var_val_final": trainer.energy_val_var[-1],
        "energy_var_val_history": trainer.energy_val_var,

        "energy_std_val_final": trainer.energy_val_var[-1]**0.5,

        "learning_rate": lr,
        "final_learning_rate": optimizer.param_groups[0]["lr"],

        "a": a,
        "training_points": training_points,

        "epochs": epochs,
        "train_loss": loss,
        "val_loss": val_loss,
        "epochs_val": epochs_val,
    }
    
    results["final_learning_rate"] = optimizer.param_groups[0]["lr"]

    with open(f"logs/{model_name}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved model to models/{model_name}.pth")
    print(f"Saved logs to logs/{model_name}.json")


def load_results(model_name):
    with open(f"logs/{model_name}.json", "r") as f:
        results = json.load(f)
    return results

def plot_loss_curves():
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

train_and_evaluate()
plot_loss_curves()
