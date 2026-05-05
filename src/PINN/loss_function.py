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

        E_L, E_K, V = self.energy_model(input_tensor) # E_L(R) =  -1/2 [∇² logpsi(R) + |∇ logpsi(R)|²] + V(R)
        residual = E_L - self.energy

        if self.model.a > 0.0:
            J = self.jastrow(input_tensor)
            residual = residual + J
            
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
        

        if self.model.dim <= 2 or float(self.model.beta) == 1.0:
            V = 0.5 * self.model.omega_ho**2 * torch.sum(positions**2, dim=(1, 2))  #dim=(1, 2) sums over N and dim
        else:
            xy_sq = torch.sum(positions[:, :, :-1]**2, dim=(1, 2))
            z_sq  = torch.sum(positions[:, :, -1]**2, dim=1)

            V = 0.5 * (
                self.model.omega_ho**2 * xy_sq +
                self.model.omega_z**2 * z_sq
            )
        
        # If interactions are included, add hard-core potential
        if self.model.a > 0.0 and self.model.N >= 2:
            eps = 1e-12
            r_ij = positions[:, :, None, :] - positions[:, None, :, :]          # (B, N, N, dim)
            r_ij_abs = torch.sqrt(torch.sum(r_ij**2, dim=-1) + eps)             # (B, N, N)
            iu = torch.triu_indices(self.model.N, self.model.N, offset=1, device=positions.device)
            pair_dist = r_ij_abs[:, iu[0], iu[1]]                               # (B, num_pairs)

            # Infinite wall: add large penalty where r_ij <= a
            inside_core = (pair_dist <= self.model.a).any(dim=1).float()        # (B,)
            V = V + inside_core * 1e6
            
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
        E_K= -0.5 * (lap_logpsi + grad_sq) 
        E_L = E_K +  V

        return E_L, E_K, V
    
    def jastrow_single(self, x):
        a   = self.model.a
        eps = 1e-12
        N   = x.shape[0]
        J   = x.new_zeros(())

        if N < 2 or a == 0.0:
            return J

        for i in range(N):
            for j in range(i + 1, N):
                r_ij = torch.sqrt(((x[i] - x[j]) ** 2).sum() + eps)
                f_ij = 1.0 - a / r_ij
                J    = J + torch.log(torch.clamp(f_ij, min=eps))
        return J