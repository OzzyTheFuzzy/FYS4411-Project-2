import copy
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset #for training data loading

from loss_function import LossFunctions
from initialize_data import InitializeData


class Training(LossFunctions):
    """
    Training class for the PINN that will solve SE
    """
    def __init__(self, SE_model, val_points=100, val_width=1.0, val_seed=17):
        super().__init__(SE_model)
        self.model  = SE_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.val_points = val_points
        self.val_width  = val_width
        self.val_seed   = val_seed

        """
        # Initialize adaptive weights for each loss component
        self.weights = {}
        self.weights["pde"] = torch.tensor(1.0, device=self.device)
        """

    def trim_dataset(self, data, num_batches):
        """
        Trims data so it's divisible by the number of batches.

        Parameters:
            - data (Tensor): Data to trim. Shape (num_points, N, dim)
            - num_batches (int): Desired number of batches.

        Returns:
            - Tensor: Trimmed data tensor
        """
        trimmed_len = (len(data) // num_batches) * num_batches

        if trimmed_len == 0:
            raise ValueError("num_batches is larger than the number of data points.")

        return data[:trimmed_len]

    def training_step_SE(self, batch_points, optimizer):

        """
        Performs a single training step: computes the PDE loss, backpropagates, and updates model parameters.
        """
        
        # Compute PDE loss
        pde_loss = self.PDE_loss(batch_points)

        # Backpropagation
        optimizer.zero_grad()
        pde_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
        optimizer.step()

        return pde_loss.item()
    
    def training_cycle_SE(self, N_epochs, positions, optimizer, scheduler, num_batches, use_scheduler=False):
        """
        Full training of the model which solves the SE
        After each batch it updates the model parameters and can update the learning rate.
        It trains for N_epochs which is the number of times the entire dataset is passed through the model

        Parameters:
            - N_epochs (int): Number of training epochs.
            - positions (torch.Tensor): Training data points of shape (num_points, N, dim).
            - optimizer (torch.optim.Optimizer): Optimizer instance.
            - scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            - num_batches (int): Number of batches per epoch.
            - use_scheduler (bool): Whether to update the learning rate scheduler after each epoch.

        Returns:
            - (tuple): Tuple containing:
                - loss (list): Training PDE loss per epoch.
                - epochs (list): Epoch indices for training.
                - val_loss (list): Validation PDE loss (evaluated every 50 epochs).
                - epochs_val (list): Epoch indices corresponding to validation losses.
                - best_model_state (dict): Model weights with lowest validation loss.
        """

        # Generate fixed validation data
        N_validation = self.val_points
        data_initializer_val = InitializeData(self.model.N, self.model.dim, device=self.device)
        val_positions = data_initializer_val.initialize_pde(N_validation, width=self.val_width, seed=self.val_seed)

        # Lists to store training and validation results
        loss, epochs = [], []
        val_loss, epochs_val = [], []

        best_val_loss = float("inf")
        best_model_state = copy.deepcopy(self.model.state_dict())

        # Trim PDE dataset to fit exact batch division.
        positions = self.trim_dataset(positions, num_batches)

        batch_size = max(1, len(positions) // num_batches) # Batch size for PDE training
        batch_array = DataLoader(TensorDataset(positions), batch_size, shuffle=True, drop_last=True) #creates batches for training

        for epoch in range(N_epochs):

            self.model.train()

            pde_sum = 0.0
            batches = 0

            for (pde_batch,) in batch_array:
                
                pde_loss = self.training_step_SE(pde_batch, optimizer) #returns float value of PDE loss for the batch
                pde_sum += pde_loss
                batches += 1


            avg_pde = pde_sum / batches # Average PDE loss for the epoch

            loss.append(avg_pde)

            epochs.append(epoch)

            # check the validation loss and save the best model state
            if epoch % 50 == 0:
                self.model.eval() # Set model to evaluation mode for validation
                pde_val_loss = self.PDE_loss(val_positions)

                val_loss.append(pde_val_loss.item())
                epochs_val.append(epoch)

                if pde_val_loss.item() < best_val_loss:
                    best_val_loss = pde_val_loss.item()
                    best_model_state = copy.deepcopy(self.model.state_dict())

            if epoch % 10 == 0:
                
                energy = self.energy_model(val_positions)
                mean_energy = energy.mean().item()
                print(f"Epoch {epoch}: Mean Energy = {mean_energy:.6f}")

            if use_scheduler:
                scheduler.step()

        return loss, epochs, val_loss, epochs_val, best_model_state

