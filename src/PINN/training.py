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
    def __init__(self, SE_model, val_points=1000, val_width=1.0, val_seed=17):
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
        self.V_coulomb_array = []
        self.energy_val_std_array =[]
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
        optimizer.zero_grad(set_to_none=True)
        pde_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
        optimizer.step()

        return pde_loss.item()
    
    def training_cycle_SE(self, N_epochs, data_initializer, training_points, width, seed, optimizer, scheduler, num_batches, coulomb_init=False, use_scheduler=False):
        """
        Full training of the model which solves the SE
        After each batch it updates the model parameters and can update the learning rate.
        It trains for N_epochs which is the number of times the entire dataset is passed through the model

        Parameters:
            - N_epochs (int): Number of training epochs.
            - data_initializer (InitializeData): Instance for initializing training data.
            - training_points (int): Number of training points.
            - width (float): Width of the Gaussian distribution for sampling training collocation points.
            - seed (int): Random seed for sampling training collocation points.
            - optimizer (torch.optim.Optimizer): Optimizer instance.
            - scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            - num_batches (int): Number of batches per epoch.
            - coulomb_init (bool): Whether to initialize the energy parameter with the mean Coulomb energy.
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
        data_initializer_val = InitializeData(self.model.N, self.model.dim, device=self.device, hard_core_radius=self.model.a, omega_z=data_initializer.omega_z)
        val_positions = data_initializer_val.initialize_pde(N_validation, width=self.val_width, seed=self.val_seed)
        
        if self.model.a > 0.0: #finetunes the initial energy for coulomb interactions
            if coulomb_init:
                self.initialize_energy_with_coulomb(val_positions)
            else: 
                self.initialize_energy_with_data()

        # Lists to store training and validation results
        loss, epochs = [], []
        val_loss, epochs_val = [], []
        
        best_E_L_var = 100000
        best_model_state = copy.deepcopy(self.model.state_dict())


        for epoch in range(N_epochs):

            positions = data_initializer.initialize_pde(training_points, width, seed= int(seed * epoch)) # Generate new collocation points for each epoch with different seed

            # Trim PDE dataset to fit exact batch division.
            positions = self.trim_dataset(positions, num_batches)
            
            batch_size = max(1, len(positions) // num_batches) # Batch size for PDE training
            batch_array = DataLoader(TensorDataset(positions), batch_size, shuffle=True, drop_last=True) #creates batches for training
                
            self.model.train()

            pde_sum = 0.0
            batches = 0

            for (pde_batch,) in batch_array:
                
                pde_loss = self.training_step_SE(pde_batch, optimizer) #returns float value of PDE loss for the batch
                pde_sum += pde_loss
                batches += 1

                del pde_batch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            avg_pde = pde_sum / batches # Average PDE loss for the epoch

            loss.append(avg_pde)
            epochs.append(epoch)

            # check the variance of the energy save the best model state
            # Monitor training and save important training data
            # Validation + monitoring
            if epoch % 10 == 0:
                self.model.eval()

                val_energy, E_K, V, V_coulomb = self.energy_model(val_positions)
                
                residual = val_energy - self.energy
                pde_val_loss_value = torch.mean(residual**2).item()
                 
                E_K_mean = E_K.mean().item()
                V_mean = V.mean().item()
                V_coulomb_mean = V_coulomb.mean().item()
                mean_energy = val_energy.mean().item()
                E_L_var = val_energy.var().item()
                E_L_std = val_energy.std().item()

                val_loss.append(pde_val_loss_value)
                epochs_val.append(epoch)

                if E_L_var < best_E_L_var:
                    best_E_L_var = E_L_var
                    best_model_state = copy.deepcopy(self.model.state_dict())

                self.E_K_array.append(E_K_mean)
                self.V_array.append(V_mean)
                self.V_coulomb_array.append(V_coulomb_mean)
                self.energy_val_mean.append(mean_energy)
                self.energy_val_var.append(E_L_var)
                self.energy_val_std_array.append(E_L_std)
                self.energy_model_array.append(self.energy.item())
                self.alpha_array.append(self.model.alpha.item())

                print(
                    f"Epoch {epoch}: <E>_val={mean_energy:.4f}: "
                    f"Var(<E>_val)={E_L_var:.4f}: "
                    f"E_K={E_K_mean:.4f}, V={V_mean:.4f}, "
                    f"V_coulomb={V_coulomb_mean:.4f}: "
                    f", alpha = {self.model.alpha.item():.4f}"
                )
                print(f"Loss = {avg_pde:.4f}, Validation Loss = {pde_val_loss_value:.4f}")
                print(f"Energy model {self.energy.item():.4f}")

                del pde_val_loss_value, val_energy, E_K, V, V_coulomb

            del positions, batch_array

            if use_scheduler:
                scheduler.step()

        return loss, epochs, val_loss, epochs_val, best_model_state



