import os
import json
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

from training import Training
from initialize_data import InitializeData
from model import Model, SE_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
width = 1.0      # Width of the Gaussian distribution for sampling collocation points
a = 0.0          # Hard-core radius (set to 0 for no interactions)
N = 2          # Number of particles (dimensions)
dim = 1         # Dimensionality of the particles

# Number of training points for each component
batch_size = 10

# Model architecture and training parameters
hidden_layers = [8, 8]
epochs = 200
num_batches = 10
model_name = "model"
N_inputs = int(N * dim)  # Number of input features (e.g., spatial coordinates)

# Create instance of data initialization 
data_initializer = InitializeData(N, dim, device=device, dtype=torch.float32, hard_core_radius=a)

# create data for training
pde = data_initializer.initialize_pde(batch_size, width, seed=17)
print(pde,pde.shape)

# Model initialization, with optimizer and scheduler
model=SE_Model(
    dim=dim,
    N=N,
    rho_hidden=[20, 20],
    phi_hidden=[20, 20],
    eta_hidden=[20, 20],
    phi_output=10,
    eta_output=10,
    activation_function=nn.Tanh(),
    alpha=0.5,
    beta=1.0,
    trainable_parameters=False)

print(model(pde))



pde.requires_grad_(True)
out = model(pde)

grad_u = torch.autograd.grad(
    outputs=out,
    inputs=pde,
    grad_outputs=torch.ones_like(out),
    create_graph=True
)[0]

print("grad_u shape:", grad_u.shape)
u1 = model(pde)
u2 = model(pde[:, [1, 0], :])
print(torch.max(torch.abs(u1 - u2)))


#for training
"""
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)



if __name__ == "__main__":
    # Training
    trainer = Training(model)
    
    loss, epochs, val_loss, epochs_val, train_components, val_components, best_model_state = trainer.training_cycle_BS(
        epochs, pde, optimizer, scheduler, num_batches, key_lbc=None, key_rbc=None
    )

    # Save model
    torch.save(best_model_state, f"/models/{model_name}.pth")

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

"""