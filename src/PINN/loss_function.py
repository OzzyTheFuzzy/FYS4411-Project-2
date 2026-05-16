import torch
import torch.nn as nn
from torch.func import jacrev
from torch.func import grad, jacrev, vmap
import numpy as np

from model import Model, SE_Model

from initialize_data import distance_and_distance_vec


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
            E_init = self.model.N * (1 + beta_val / 2) 
        

        if self.model.trainable_energy:
            self.energy = nn.Parameter(torch.tensor(E_init, device=self.device))
        else:
            self.register_buffer("energy", torch.tensor(E_init, device=self.device))

        self.mse    = nn.MSELoss()

    def initialize_energy_with_coulomb(self, positions):
        beta_val = float(self.model.beta)

        if beta_val == 1.0:
            E_init = self.model.N * self.model.dim / 2
        else:
            E_init = self.model.N * (1 + beta_val / 2)

        if self.model.N >= 2 and self.model.a > 0.0:
            with torch.no_grad():
                V_coulomb_mean = self.coulomb(positions).mean()
            E_init = E_init + V_coulomb_mean.item()

        self.energy.data = torch.tensor(
            E_init,
            device=self.device,
            dtype=self.energy.dtype
        )
    def initialize_energy_with_VMC(self):

        if self.model.N ==2 and self.model.beta!=1.0:
            E_VMC = 2.8341

            self.energy.data = torch.tensor(
            E_VMC,
            device=self.device,
            dtype=self.energy.dtype)

        if self.model.N==2 and self.model.beta==1.0 and self.model.a > 0.0 and self.model.dim==2:
            E_VMC = 3.0
            self.energy.data = torch.tensor(
            E_VMC,
            device=self.device,
            dtype=self.energy.dtype)

        if self.model.N==10 and self.model.dim==3 and self.model.a > 0.0 and self.model.beta!=1.0:
            E_VMC = 24.405  # obtained from VMC sampling with the initial model, can be adjusted based on the specific system and model initialization
            
            self.energy.data = torch.tensor(
                E_VMC,
                device=self.device,
                dtype=self.energy.dtype
            )

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

        E_L = self.energy_model(input_tensor, total=True) # E_L(R) =  -1/2 [∇² logpsi(R) + |∇ logpsi(R)|²] + V(R)
        residual = E_L - self.energy
            
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
            """function to compute logpsi for a single position x of shape (N, dim)"""
            nn_out = self.model(x.unsqueeze(0))[0, 0]

            return nn_out            

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
        #symmetric potential
        if self.model.dim <= 2 or float(self.model.beta) == 1.0:
            V = 0.5 * self.model.omega_ho**2 * torch.sum(positions**2, dim=(1, 2))  #dim=(1, 2) sums over N and dim
        #cylindrical potential
        else:
            xy_sq = torch.sum(positions[:, :, :-1]**2, dim=(1, 2))
            z_sq  = torch.sum(positions[:, :, -1]**2, dim=1)

            V = 0.5 * (
                self.model.omega_ho**2 * xy_sq +
                self.model.omega_z**2 * z_sq
            )

        return V.unsqueeze(1)

    def coulomb(self, positions):
        """
        positions: (B, N, dim)
        returns V_coulomb: (B, 1)
        """
        r_ij_abs, r_ij= distance_and_distance_vec(positions)   # (B, N, N), (B, N, N, dim)
        iu = torch.triu_indices(self.model.N, self.model.N, offset=1, device=positions.device)

        pair_dist = r_ij_abs[:, iu[0], iu[1]]  
  
        eps=1e-8
        V_coulomb = self.model.a * torch.sum(1.0 / (pair_dist + eps), dim=1, keepdim=True)  # shape (B, 1)
   
        return V_coulomb
    
    def energy_model(self, positions, total=False):
        """
        positions: (B, N, dim)
        returns E_L: (B, 1)
       
        Calculates: E_L(R) =  -1/2 [∇² logpsi(R) + |∇ logpsi(R)|²] + V(R)
       
        """
        
        logpsi, grad_logpsi, lap_logpsi = self.logpsi_grad_laplacian(positions)
        #potential energy
        V = self.potential(positions)

        grad_sq = torch.sum(grad_logpsi**2, dim=(1, 2)).unsqueeze(1)

        E_K = -0.5 * (lap_logpsi + grad_sq) 
        
        if self.model.N >1 and self.model.a > 0.0:
            V_coulomb = self.coulomb(positions)
        else:
            V_coulomb = torch.zeros_like(V)

        #calculate the local energy E_L for each position in the batch 
        
        E_L = E_K + V + V_coulomb         # size (B, 1)

        if total: #shortcut for memory saving
            return E_L
        
        else:
            return E_L, E_K, V, V_coulomb
    
