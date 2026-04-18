import torch
import torch.nn as nn
from torch.func import jacrev
from torch import vmap
import numpy as np

from model import Model
from initialize_data import InitializeData


class LossFunctions():
    """
    Collection of loss function used to train a Physics-Informed Neural Network (PINN)
    for solving high-dimensional Black-Scholes-type PDEs.
    """
    def __init__(self, model, S_max_list, r=1.0, K=1.0, sigma=1.0, T=1.0, rho=None):
        """
        Initializes loss functions for PDE training.

        Parameters:
            - model          (nn.Module): Neural network representing the PDE solution.
            - S_max_list (list of float): Max values for each asset.
            - r                  (float): Risk-free interest rate.
            - K                  (float): Strike price of the option.
            - sigma      (list of float): Volatility for each asset.
            - T                  (float): Time to maturity.
            - rho    (tensor or ndarray): Correlation matrix (defaults to identity).
        """
        self.model = model
        self.S_max_list = S_max_list
        self.r = r
        self.K = K
        self.T = T
        self.d = len(S_max_list)  # Number of assets
        self.global_max = np.max(S_max_list) 
        self.mse = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sigma = torch.as_tensor(sigma, dtype=torch.float32, device=self.device).flatten()  # Make sure sigma is a tensor and on correct device

        # Use identity matrix if no correlation matrix is given
        if rho is None:
            self.rho = torch.eye(self.d, dtype=torch.float32, device=self.device)
        else:
            self.rho = torch.as_tensor(rho, dtype=torch.float32, device=self.device)



    def PDE_loss(self, input_tensor):
        """
        Computes the mean squared residual of the PDE.

        Parameters:
            - input_tensor (torch.Tensor): Normalized inputs (S_norm, t_norm)
        
        Returns:
            - torch.Tensor: Scalar PDE residual loss.
        """
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        x_norm = input_tensor[:, :self.d] 
        tau_norm = input_tensor[:, -1:] 
       
        X = torch.cat([x_norm, tau_norm], dim=1)
        X = X.float().requires_grad_()
   
        # Model prediction
        C = self.model(X)

        # Compute gradients
        dC = torch.autograd.grad(C, X, torch.ones_like(C), create_graph=True)[0]
        C_tau_norm = dC[:, -1:]  # ∂C/∂τ
        C_x_norm = dC[:, :self.d]  # ∇C with respect to each asset
        
        # Denormalize asset prices and time
        S = 0.5 * (x_norm + 1.0) * self.global_max
        C_t = C_tau_norm * (-2.0 / self.T)   

        if self.d == 1: 
            # Compute second derivative with respect to asset
            C_xx_norm = torch.autograd.grad(C_x_norm, x_norm, torch.ones_like(C_x_norm), create_graph=True)[0]

            # Denormalize first and second derivatives
            C_s = C_x_norm * (2.0 / self.S_max_list[0]) 
            C_ss = C_xx_norm * ((2.0 / self.S_max_list[0])**2)

            # Compute PDE residual
            pde_residual = C_t + (0.5 * self.sigma[0]**2 * S**2 * C_ss) + (self.r * C_s * S) - (self.r * C)

            return torch.mean(pde_residual**2)


        else:
            _, self.d = x_norm.shape

            # Define a function that returns the spatial gradient for one sample
            def grad_C_x(x, tau):
                xt = torch.cat([x, tau], dim=-1).unsqueeze(0)  # [1, d+1]
                C_val = self.model(xt)  # [1, 1]
                dC_dX = torch.autograd.grad(C_val, xt, torch.ones_like(C_val), create_graph=True)[0]  # ∂C/∂X  → [1, d+1]
                return dC_dX[0, :self.d]  # [d]

            #Compute Hessian using Jacobian of the gradient
            hessian_single = jacrev(grad_C_x, argnums=0)
            H_xx_norm = vmap(hessian_single, in_dims=(0, 0))(x_norm, tau_norm)

            # Convert normalized second derivatives to original coodinates
            scales = (2.0 / torch.tensor(self.S_max_list, device=x_norm.device))
            scale_mat = scales[None, :] * scales[:, None]  # [d, d]
            H_SS = H_xx_norm * scale_mat.unsqueeze(0)  # [batch, d, d]

            # Denormalize first derivatives
            C_s = C_x_norm * (2.0 / torch.tensor(self.S_max_list, device=self.device))  # [batch, d]

            # Initialize PDE residual with time derivatives
            pde_residual = C_t.clone()  # [batch, 1]

            # Add volatility and correlation terms
            for i in range(self.d):
                for j in range(self.d):
                    pde_residual += (
                        0.5
                        * self.sigma[i]
                        * self.sigma[j]
                        * self.rho[i, j]
                        * S[:, i:i+1]
                        * S[:, j:j+1]
                        * H_SS[:, i, j].unsqueeze(-1)
                    )

            # Add drift (first term) and discounting terms (second term)
            drift = (self.r * S * C_s).sum(dim=1, keepdim=True)
            pde_residual += drift - (self.r * C)

            return torch.mean(pde_residual**2)
