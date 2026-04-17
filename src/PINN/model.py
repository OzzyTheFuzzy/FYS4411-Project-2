#model.py
import torch
import torch.nn as nn
import torch.nn.init as init

class Model(nn.Module):
    """
    A fully connected feed-forward neural network designed for solving PDEs using 
    Physics-Informed Neural Networks (PINNs). 

    This model allows customization of the input size, the number and size of hidden layers, and the activation function.
    """
    def __init__(self, num_inputs, hidden_layers, activation_function):
        """
        Initializes the fully connected neural network.

        Parameters:
            - num_inputs                (int): Number of input features (e.g., spatial and temporal coordinates).
            - hidden_layers     (list of int): List specifying the number of neurons in each hidden layer.
            - activation_function (nn.Module): Activation function to apply after each hidden layer.
        """
        super(Model, self).__init__()  # Initialize the parent class

        self.activation = activation_function
        
        # Build the network layer by layer
        layers = [nn.Linear(num_inputs, hidden_layers[0]), self.activation]
        for i in range(1, len(hidden_layers)):
            layers += [nn.Linear(hidden_layers[i-1], hidden_layers[i]), self.activation]
        layers.append(nn.Linear(hidden_layers[-1], 1))


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

model = Model(num_inputs=2, hidden_layers=[20,20,20,20], activation_function=nn.Tanh())

print(model(torch.tensor([[3.0, 3.0],[2.0, 2.0]])))
