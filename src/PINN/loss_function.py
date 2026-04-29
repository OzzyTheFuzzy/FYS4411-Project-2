import torch
import torch.nn as nn
from torch.func import jacrev
from torch.func import grad, jacrev, vmap
import numpy as np

from model import Model, SE_Model

from initialize_data import InitializeData


class LossFunctions(nn.Module):
    """
    Collection of loss function used to train a Physics-Informed Neural Network (PINN)
    for solving high-dimensional Black-Scholes-type PDEs.
    """
    def __init__(self, SE_model):
        super().__init__()
        
        #Initializes loss function, model and energy parameter for training

        self.model  = SE_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
        beta_val = float(self.model.beta) # Convert to float if it's a tensor
        # Calculate initial energy based on dim, N and beta_val
        if beta_val == 1.0:
            E_init = self.model.N * self.model.dim / 2
        else:
            E_init = self.model.N * (1 + np.sqrt(beta_val) / 2) 

        self.energy = nn.Parameter(torch.tensor(E_init, device=self.device))
        self.mse    = nn.MSELoss()


    def PDE_loss(self, positions):
        """
        # Change the energy!
        Computes the mean squared residual of the PDE.

        Parameters:
            - input_tensor (torch.Tensor): ())
        
        Returns:
            - torch.Tensor: Scalar PDE residual loss.
        """

        input_tensor = positions.clone().requires_grad_(True)

        E_loc = self.energy_model(input_tensor) # E_L(R) =  -1/2 [∇² logpsi(R) + |∇ logpsi(R)|²] + V(R)
        residual = E_loc - self.energy
       
        return torch.mean(residual**2)
        
       
    
    def logpsi_grad_laplacian(self, positions):
        """
        positions: (B, N, dim)
        returns:
            logpsi: (B, 1)
            grad_logpsi: (B, N, dim)
            lap_logpsi: (B, 1)
        """

        def logpsi_single(x):
            return self.model(x.unsqueeze(0))[0, 0]

        grad_fn = grad(logpsi_single)
        hess_fn = jacrev(grad_fn)

        grad_logpsi = vmap(grad_fn)(positions)      # (B, N, dim)
        hess_logpsi = vmap(hess_fn)(positions)      # (B, N, dim, N, dim)

        B = positions.shape[0]
        D = self.model.N * self.model.dim

        hess_flat = hess_logpsi.reshape(B, D, D)
        lap_logpsi = hess_flat.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True)

        logpsi = self.model(positions)

        return logpsi, grad_logpsi, lap_logpsi
    
    def potential(self, positions):
        """
        positions: (B, N, dim)
        returns V: (B, 1)
        """
        

        if self.model.dim <= 2 or float(self.model.beta) == 1.0:
            V = 0.5 * self.model.omega_ho**2 * torch.sum(positions**2, dim=(1, 2))  #dim=(1, 2) sums over N and dim
        else:
            xy_sq = torch.sum(positions[:, :, :-1]**2, dim=(1, 2))
            z_sq  = torch.sum(positions[:, :, -1]**2, dim=1)

            V = 0.5 * (
                self.model.omega_ho**2 * xy_sq +
                self.model.omega_z**2 * z_sq
            )

        return V.unsqueeze(1)
    
    def energy_model(self, positions):
        """
        positions: (B, N, dim)
        returns E_L: (B, 1)
       
        Calculates: E_L(R) =  -1/2 [∇² logpsi(R) + |∇ logpsi(R)|²] + V(R)
       
        """
        logpsi, grad_logpsi, lap_logpsi = self.logpsi_grad_laplacian(positions)
        V = self.potential(positions)

        grad_sq = torch.sum(grad_logpsi**2, dim=(1, 2)).unsqueeze(1)

        #calculate the local energy E_L for each position in the batch 
        E_L = -0.5 * (lap_logpsi + grad_sq) + V 

        return E_L
    