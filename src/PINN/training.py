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
        self.energy_val_var = [] # to store the variance of the energy on the validation set during training
        self.energy_val_mean = [] # to store the mean of the energy on the validation set during training
        self.energy_model_array = [] # to store the value of the energy parameter during training
        self.alpha_array = [] # to store the value of alpha during training
        self.E_K_array = []
        self.V_array = []
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
        
        best_E_L_var = 100000
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

            # check the variance of the energy save the best model state
            if epoch % 10 == 0:
                self.model.eval()

                pde_val_loss = self.PDE_loss(val_positions)
                E_L, E_K, V = self.energy_model(val_positions)

                E_L_var = E_L.var().item()

                val_loss.append(pde_val_loss.item())
                epochs_val.append(epoch)

                if E_L_var < best_E_L_var:
                    best_E_L_var = E_L_var
                    best_model_state = copy.deepcopy(self.model.state_dict())
            
            # Monitor training and save important training data
            if epoch % 10 == 0:
                # for monitoring training progress, 
                # we compute the energy on the validation set and store its mean and variance
                # and the parameters alpha and E_model for the current epoch
                val_energy, E_K, V = self.energy_model(val_positions)

                E_K_mean= E_K.mean().item()
                self.E_K_array.append(E_K_mean)

                V_mean = V.mean().item()
                self.V_array.append(V_mean)

                mean_energy = val_energy.mean().item()
                self.energy_val_mean.append(mean_energy)

                E_L_var = val_energy.var().item()
                self.energy_val_var.append(E_L_var)
               
                energy_model_value = self.energy.item()
                self.energy_model_array.append(energy_model_value)
                
                self.alpha_array.append(self.model.alpha.item())

                print(f"Epoch {epoch}: <E>_val={mean_energy:.8f}: E_K={E_K_mean:.8f}, V={V_mean:.8f}: Var(<E>_val)={E_L_var:.8f}, alpha = {self.model.alpha.item():.4f}")
                print(f"Loss = {avg_pde:.8f}, Validation Loss = {pde_val_loss.item():.8f}")
                print(f'Energy model {self.energy.item():.8f}')

            if use_scheduler:
                scheduler.step()

        return loss, epochs, val_loss, epochs_val, best_model_state

