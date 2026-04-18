import os
import json
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

from training import Training
from initialize_data import InitializeData
from model import Model 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory to store results
test_dir = "test_results"
os.makedirs(test_dir, exist_ok=True)
os.makedirs(f"{test_dir}/models", exist_ok=True)
os.makedirs(f"{test_dir}/logs", exist_ok=True)


# Configuration
S_max_list = [20]   # Maximum asset price for each asset 
sigma_list = [0.2]  # Volatility
T = 1.0             # Maturity time
r = 0.05            # Risk-free rate
K = 3.0             # Strike price


# Number of training points for each component
lbc_N = 1500        # Left boundary condition
rbc_N = 150         # Right coundary condition
tc_N = 150          # Terminal condition
tc_N_extra = 500    # Extra points for the terminal condition
pde_N = 150         # Interior collocation points


# Model architecture and training parameters
hidden_layers = [32]
epochs = 300
num_batches = 10
model_name = "model"


# Data initialization
data_initializer = InitializeData(S_max_list, T)
lbc = data_initializer.initialize_bc(lbc_N, seed=1)
rbc = data_initializer.initialize_bc(rbc_N, seed=3, rbc=True)
tc = data_initializer.initialize_tc(tc_N, tc_N_extra, seed=7)
pde = data_initializer.initialize_pde(pde_N, 0, seed=12)

# Model initialization, with optimizer and scheduler
model = Model(len(S_max_list)+1, hidden_layers, nn.Tanh())
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)



if __name__ == "__main__":
    # Training
    trainer = Training(model, S_max_list, r, K, sigma_list, T)
    
    loss, epochs, val_loss, epochs_val, train_components, val_components, best_model_state = trainer.training_cycle_BS(
        epochs, lbc, rbc, tc, pde, optimizer, scheduler, num_batches, key_lbc=None, key_rbc=None
    )

    # Save model
    torch.save(best_model_state, f"{test_dir}/models/{model_name}.pth")

    # Save training results to JSON
    results = {
        "epochs": epochs,
        "train_loss": loss,      
        "val_loss": val_loss,     
        "min_val_loss": float(min(val_loss)), 
        "train_components": train_components,
        "val_components": val_components,
    }

    with open(f"{test_dir}/logs/{model_name}" + ".json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved all results to {test_dir}/logs/{model_name}.json")


    # Plot loss curve
    plt.plot(results["epochs"], loss, label="Training")
    plt.plot(results["epochs"][::50], val_loss, label="Validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()