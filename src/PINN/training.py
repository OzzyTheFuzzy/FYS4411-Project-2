import copy
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from loss_function import LossFunctions
from initialize_data import InitializeData


class Training(LossFunctions):
    """
    Training class for Physics-Informed Neural Networks (PINNs) applied to high-dimensional
    Black-Scholes-type PDEs. Inherits form the LossFunction class, which provides the
    loss definition for the terminal condition, boundary conditions, and PDE residual.
    """
    def __init__(self, model, S_max_list, r=1, K=1, sigma=1, T=1):
        """
        Initialize the Training class for PINN-based Black-Scholes training.

        Parameters:
            - model          (nn.Module): The neural network model.
            - S_max_list (list of float): Max asset values for each dimension.
            - r                  (float): Risk-free interest rate.
            - K                  (float): Strike price.
            - sigma      (list of float): Volatility per asset.
            - T                  (float): Time to maturity.
        """
        super().__init__(model, S_max_list, r, K, sigma, T)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize adaptive weights for each loss component
        self.weights = {}
        for i in range(len(S_max_list)):
            self.weights[f"bc_left_{i}"] = torch.tensor(1.0, device=self.device)
            self.weights[f"bc_right_{i}"] = torch.tensor(1.0, device=self.device)
        self.weights["tc"] = torch.tensor(1.0, device=self.device)
        self.weights["pde"] = torch.tensor(1.0, device=self.device)
        

    
    def training_step_BS(self, lbc, rbc, tc, pde, optimizer, key_lbc, key_rbc):
        """
        Performs a single training step: computes all loss components, adapts weights,
        backpropagates the weighted loss, and updates model parameters.

        Parameters:
            - lbc                      (Tensor): Left boundary condition data (d, N, d+1).
            - rbc                      (Tensor): Right boundary condition data (d, N, d+1).
            - tc                       (Tensor): Terminal condition data (N, d+1).
            - pde                      (Tensor): PDE interior sampling points (N, d+1).
            - optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            - key_lbc                     (int): Index of asset dimension for left boundary loss.
            - key_rbc                     (int): Index of asset dimension for right boundary loss.
        
        Returns:
            - (tuple): Tuple containing:
                - total_loss    (Tensor): Total weighted loss.
                - pde_loss      (Tensor): PDE residual loss.

        """
        # Compute each component of the loss

        pde_loss = self.PDE_loss(pde)
            

        raw_losses =pde_loss

        # Adaptive weighting based on the total gradient (GradNorm)
        epsilon, alpha = 1e-8, 0.9
        grads = {}
            

        optimizer.zero_grad()

        # Get gradients of model parameters with respect to current loss
        param_grads = torch.autograd.grad(
            pde_loss,
            self.model.parameters(),
            retain_graph=True
        )

        # Compute total gradient norm (L2 norm)
        grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in param_grads if g is not None))
        
        
        #test to check for vanishing gradients 
        if grad_norm < 1e-8:
                print(f"[WARNING] Vanishing gradient detected ")
            
        total_grad = sum(grads.values())
                        
        # Compute relative weights based on inverse gradient magnitudes
        #weights_hat = {n: total_grad / (grads[n] + epsilon) for n in loss_names}
        
        #self.weights = alpha * self.weights[n] + (1 - alpha) * weights_hat[n]  # Update rule: moving average

        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

        # Compute weighted losses using updated weights
        weighted_losses = self.weights * raw_losses 
        
        total_loss = sum(weighted_losses.values())

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
        optimizer.step()

        return total_loss, weighted_losses["pde"], weighted_losses["tc"]
    


    def training_cycle_BS(self, N_epochs, pde, optimizer, scheduler, num_batches, key_lbc, key_rbc, use_scheduler=False):
        """
        Full training cycle using only PDE residual loss.

        Parameters:
            - N_epochs                       (int): Number of training epochs.
            - pde                         (Tensor): PDE interior point training data.
            - optimizer    (torch.optim.Optimizer): Optimizer instance.
            - scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            - num_batches                    (int): Number of batches per epoch.
            - use_scheduler                 (bool): Whether to use lr scheduler after each epoch.

        Returns:
            - (tuple): Tuple containing:
                - train_loss       (list): PDE training loss per epoch.
                - train_epochs     (list): Epoch indices for training.
                - val_loss         (list): PDE validation loss (sampled every 50 epochs).
                - val_epochs       (list): Epochs corresponding to validation losses.
                - train_components (dict): Training loss components by type and epoch.
                - val_components   (dict): Validation loss components by type and epoch.
                - best_model_state (dict): Model weights with lowest validation loss.
        """
        # Keep only PDE terms in component tracking.
        train_components = {k: [] for k in ["total", "pde"]}
        val_components = {k: [] for k in ["total", "pde"]}

        # Lists to store training and validation results
        loss, epochs = [], []
        val_loss, epochs_val = [], []

        best_val_loss = float("inf")
        best_model_state = copy.deepcopy(self.model.state_dict())

        # Trim PDE dataset to fit exact batch division.
        usable_n = (len(pde) // num_batches) * num_batches
        if usable_n == 0:
            raise ValueError("num_batches is larger than the number of PDE points.")
        pde = pde[:usable_n]

        bs_pde = max(1, len(pde) // num_batches)
        pde_dl = DataLoader(TensorDataset(pde), bs_pde, shuffle=True, drop_last=True)

        # Use the full PDE set as a fixed validation reference.
        pde_val = pde.clone()

        for epoch in range(N_epochs):
            self.model.train()

            pde_sum = 0.0
            batches = 0

            for (pde_batch,) in pde_dl:
                optimizer.zero_grad()
                pde_loss = self.PDE_loss(pde_batch)
                pde_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                optimizer.step()

                pde_sum += pde_loss.item()
                batches += 1

            avg_pde = pde_sum / batches

            loss.append(avg_pde)
            epochs.append(epoch)
            train_components["total"].append(avg_pde)
            train_components["pde"].append(avg_pde)

            if epoch % 50 == 0:
                self.model.eval()
                pde_val_loss = self.PDE_loss(pde_val)

                val_loss.append(pde_val_loss.item())
                epochs_val.append(epoch)
                val_components["total"].append(pde_val_loss.item())
                val_components["pde"].append(pde_val_loss.item())

                if pde_val_loss.item() < best_val_loss:
                    best_val_loss = pde_val_loss.item()
                    best_model_state = copy.deepcopy(self.model.state_dict())

            if use_scheduler:
                scheduler.step()

            if epoch % 100 == 0:
                if val_loss:
                    print(f"T: Epoch {epoch}, pde {avg_pde:.3e}, loss {avg_pde:.3e}")
                    print(f"V: Epoch {epoch}, pde {val_loss[-1]:.3e}, loss {val_loss[-1]:.3e}")
                else:
                    print(f"T: Epoch {epoch}, pde {avg_pde:.3e}, loss {avg_pde:.3e}")

        return loss, epochs, val_loss, epochs_val, train_components, val_components, best_model_state