import os
import json
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

from training import Training
from initialize_data import InitializeData
from model import Model, SE_Model

model_name  = "model_test"

def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration
    width = 1.0      # Width of the Gaussian distribution for sampling collocation points
    a = 0.0          # Hard-core radius (set to 0 for no interactions)
    N = 1          # Number of particles (dimensions)
    dim = 1         # Dimensionality of the particles

    # Number of training points 
    training_points = 10000

    #  Training parameters
    epochs      = 1000
    num_batches = 10
    
    N_inputs    = int(N * dim)  # Number of input features (e.g., spatial coordinates)
    val_points  = 10000
    val_width   = 1.0 
    val_seed    = 42
    lr          = 1e-3
  

    # Create instance of data initialization 
    data_initializer = InitializeData(N, dim, device=device, dtype=torch.float32, hard_core_radius=a)

    # create data for training
    positions = data_initializer.initialize_pde(training_points, width, seed=17)

    # Model initialization, with optimizer and scheduler
    model = SE_Model(
        dim=dim,
        N=N,
        rho_hidden=[32, 32],
        phi_hidden=[32, 32],
        eta_hidden=[32, 32],
        phi_output=10,
        eta_output=10,
        activation_function=nn.Tanh(),
        alpha=0.5,
        beta=1.0,
        trainable_alpha=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # Create an instance of the training class
    trainer = Training(model, val_points, val_width, val_seed)

    # train the model and get training results
    loss, epochs, val_loss, epochs_val, best_model_state = trainer.training_cycle_SE(
            
            N_epochs=epochs,
            positions=positions,
            optimizer=optimizer,
            scheduler=scheduler,
            num_batches=num_batches,
            use_scheduler=True
        )

    # Save model
    torch.save(best_model_state, f"models/{model_name}.pth")

    # Save training results to JSON
    results = {
        "epochs": epochs,
        "train_loss": loss,      
        "val_loss": val_loss,  
        "epochs_val": epochs_val,
        "min_val_loss": float(min(val_loss)),
        "min_val_epoch": int(epochs_val[val_loss.index(min(val_loss))])   }

    with open(f"logs/{model_name}" + ".json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved all results to logs/{model_name}.json")


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