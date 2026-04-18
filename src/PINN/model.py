#model.py
from numpy import shape
import torch
import torch.nn as nn
import torch.nn.init as init

class Model(nn.Module):
    """
    Class for making generic NN models
    This model allows customization of the input size, the number and size of hidden layers, and the activation function.
    """
    def __init__(self, num_inputs, num_outputs, hidden_layers, activation_function):
        """
        Initializes the fully connected neural network.

        Parameters:
            - num_inputs                (int): Number of input features (e.g., spatial and temporal coordinates).
            - num_outputs               (int): Number of output features.
            - hidden_layers     (list of int): List specifying the number of neurons in each hidden layer.
            - activation_function (nn.Module): Activation function to apply after each hidden layer.
        """
        super(Model, self).__init__()  # Initialize the parent class

        self.activation = activation_function
        
        # Build the network layer by layer
        layers = [nn.Linear(num_inputs, hidden_layers[0]), self.activation]
        for i in range(1, len(hidden_layers)):
            layers += [nn.Linear(hidden_layers[i-1], hidden_layers[i]), self.activation]
        layers.append(nn.Linear(hidden_layers[-1], num_outputs))


        # Create model and move it to the appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(*layers).to(device)

        # Initialize weights using Xavier initialization 
        self.model.apply(self._init_weights)

       

    def _init_weights(self, m):
        """
        Applies Xavier unifeorm initialization to the weights of linear layers,
        and sets their biases to zero.

        Parameters:
            - m (nn.Module): A layer in the model passed during model.apply().
        """
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight, gain=init.calculate_gain('tanh'))
            if m.bias is not None:
                init.zeros_(m.bias)  # Set bias to zero
        
     
    
    def forward(self, x):
        """
        Performs the forward pass of the neural network.

        Parameters:
            - x (torch.Tensor): Input tensor of shape (batch_size, num_inputs),
                                typically representing spatial and/or temporal coordinates.

        Returns:
            - torch.Tensor:     Output tensor containing the model's predictions.
        """       
        # Pass input through the network
        return self.model(x)

#model = Model(num_inputs=2, hidden_layers=[20,20,20,20], activation_function=nn.Tanh())
#print(model(torch.tensor([[3.0, 3.0],[2.0, 2.0]])))




class SE_Model(nn.Module):
    """
    Class for specific model with Gaussian envelope and two branches 
    (one-body and pairwise) for the wavefunction approximation.
    Parameters:
        - dim (int): Dimensionality of the particle positions (e.g., 2 for 2D).
        - N (int): Number of particles.
        - rho_hidden (list of int): Hidden layer sizes for the rho network.
        - phi_hidden (list of int): Hidden layer sizes for the phi network.
        - eta_hidden (list of int): Hidden layer sizes for the eta network.
        - phi_output (int): Output size of the phi network.
        - eta_output (int): Output size of the eta network.
        - activation_function (nn.Module): Activation function to use in all networks.
        - alpha (float): Coefficient for the Gaussian envelope term.
    """
    def __init__(
        self,
        dim,
        N,
        rho_hidden,
        phi_hidden,
        eta_hidden,
        phi_output,
        eta_output,
        activation_function=nn.Tanh(),
        alpha=0.5,
        beta=1.0,
        trainable_parameters=False
    ):
        super().__init__()

        self.dim = dim
        self.N = N
        self.alpha = alpha
        self.beta = beta

        if trainable_parameters:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))

        # returns vector
        self.phi = Model(
            num_inputs=dim,
            num_outputs=phi_output,
            hidden_layers=phi_hidden,
            activation_function=activation_function
        )
        # returns vector
        self.eta = Model(
            num_inputs=1,
            num_outputs=eta_output,
            hidden_layers=eta_hidden,
            activation_function=activation_function
        )

        # the actual network which we train. Returns scalar
        self.rho = Model(
            num_inputs=phi_output + eta_output,
            num_outputs=1,
            hidden_layers=rho_hidden,
            activation_function=activation_function
        )

    def forward(self, positions):

        """
        Computes the log-wavefunction u_theta for a batch of particle configurations.
        Parameters:
            - positions (torch.Tensor): Tensor of shape (batch_size, N, dim) 
        """
        
        B = positions.shape[0]
        x = positions  # shape (B, N, dim)

        # Gaussian envelope term
        if self.beta == 1.0:
            gauss = -(self.alpha * (x**2).sum(dim=2)).sum(dim=1, keepdim=True)  # shape (B, 1)
        
        else:
            gauss = -(self.alpha * (x[:, :, 0]**2 + x[:, :, 1]**2) 
                      + self.beta  * (x[:, :, 2]**2)).sum(dim=1, keepdim=True)   #shape (B, 1, 1)
    
        # One-body branch: sum_i phi(x_i)
        x_flat = x.reshape(B * self.N, self.dim)      # shape (B*N, dim)
        phi_out = self.phi(x_flat)                    # shape (B*N, phi_output)
        phi_out = phi_out.view(B, self.N, -1)         # shape (B, N, phi_output)
        phi_sum = phi_out.sum(dim=1)                  # shape (B, phi_output)

        # Pairwise branch: sum_{i<j} eta(r_ij)
        if self.N < 2:
            eta_sum = torch.zeros(
                B,
                self.eta.model[-1].out_features,
                device=x.device,
                dtype=x.dtype
            )
        else:
            # Pairwise displacement vectors: (B, N, N, dim)
            r_ij = x[:, :, None, :] - x[:, None, :, :]

            # Pairwise distances: (B, N, N)
            r_ij_abs = torch.linalg.norm(r_ij, dim=-1)

            # Extract unique pairs i < j
            iu = torch.triu_indices(self.N, self.N, offset=1, device=x.device)
            pair_dist = r_ij_abs[:, iu[0], iu[1]]      # shape (B, num_pairs)

            # Feed distances through eta
            pair_dist_flat = pair_dist.reshape(-1, 1)  # shape (B*num_pairs, 1)
            eta_out = self.eta(pair_dist_flat)         # shape (B*num_pairs, eta_output)

            num_pairs = pair_dist.shape[1]
            eta_out = eta_out.view(B, num_pairs, -1)   # shape (B, num_pairs, eta_output)
            eta_sum = eta_out.sum(dim=1)               # shape (B, eta_output)

        # Combine both branches and apply rho
        
        combined = torch.cat([phi_sum, eta_sum], dim=1)  # shape (B, phi_output + eta_output)
        correction = self.rho(combined)                  # shape (B, 1)

        # Final log-wavefunction
        u_theta = gauss + correction

        return u_theta 
    

"""
model=SE_Model(
    dim=2,
    N=3,
    rho_hidden=[20, 20],
    phi_hidden=[20, 20],
    eta_hidden=[20, 20],
    phi_output=10,
    eta_output=10,
    activation_function=nn.Tanh(),
    alpha=0.5,
    beta=1.0,
    trainable_parameters=False)
"""

