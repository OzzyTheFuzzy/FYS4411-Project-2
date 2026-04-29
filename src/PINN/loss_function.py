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
        
        u, lap_u = self.grad_and_laplacian(input_tensor)
        V = self.potential(input_tensor)

        residual = -0.5 * lap_u + V * u - self.energy * u

        return torch.mean(residual**2)
        
       
    
    def grad_and_laplacian(self, positions):
        """
        positions: (B, N, dim)
        returns:
            - u: (B, 1)
            - grad_u: (B, N, dim)
            - lap_u: (B, 1)
        """

        # function requires input of shape (N, dim) and returns scalar output(for hess, lap_u)
        def u_single(x):
            return self.model(x.unsqueeze(0))[0, 0] #returns scalar

        grad_u_fn = grad(u_single)
        hess_u_fn = jacrev(grad_u_fn)

        #grad_u = vmap(grad_u_fn)(positions)         # (B, N, dim)
        hess_u = vmap(hess_u_fn)(positions)         # (B, N, dim, N, dim)

        B = positions.shape[0]

        hess_flat = hess_u.reshape(B, self.model.N * self.model.dim, self.model.N * self.model.dim)
        lap_u = hess_flat.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True)

        u = self.model(positions)

        return u, lap_u
    
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

        u, lap_u = self.grad_and_laplacian(positions)
        V = self.potential(positions)

        eps = 1e-8
        return (-0.5 * lap_u + V * u) / (u + eps)