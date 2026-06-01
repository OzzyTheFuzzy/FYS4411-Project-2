# Configuration file for experimentation.py
import torch
import torch.nn as nn

# Configuration parameters
width = 1.33     # Width of the Gaussian distribution for sampling collocation points
a     = 1.0     # a=1.0  for strength of the Coulomb interactions   
N     = 10       # Number of particles (dimensions)
dim   = 3        # Dimensionality of the particles
beta  = 2.82843  #set beta=1.0 for isotropic case and beta=2.82843 for antisotropic case

omega_z  = beta       # Frequency of the harmonic trap in the z-direction. Set equal to beta for antisotropic case, and to 1 for isotropic cas
beta_jastrow = 0.5 # Wavefunction parameter when interactions are included 
alpha    = 0.5      # Wavefunction parameter for the single-particle part of the wavefunction
omega_ho = 1.0        # Frequency of the harmonic trap in the x and y directions

#  Training parameters
trainable_energy = False # whether to train the energy parameter or keep it fixed during training
initialize_gaussian = True #initialize training points with Gaussian set True. Set False for ellipsoid shell initialization

training_points = 2000
seed        = 17
epochs      = 500
batch_size  = 50
num_batches = training_points // batch_size
val_points  = 2000
val_width   = width # width of the Gaussian distribution for sampling validation collocation points
val_seed    = 42    # random seed for sampling validation collocation points
gamma       = 0.995 # learning rate decay factor for scheduler, set to a value close to 1 for smoother convergence
lr          = 5e-3 # learning rate for optimizer. Will be tuned during training by scheduler for smoother convergence
lr_E        = 1e-3 # learning rate for energy parameter, set lower than lr for smoother convergence towards true GS energy
lr_alpha    = 1e-6 # learning rate for alpha parameter, set lower than
trainable_alpha = False # whether to train the energy parameter alpha or keep it fixed during training
coulomb_init    = False # if we do not have hard coded energy results for the given config, use coulomb initialization
trainable_beta_jastrow = False # whether to train the beta_jastrow parameter or keep it fixed during training
lr_beta_jastrow = 1e-5

model_config = {
        "dim": dim,
        "N": N,
        "rho_hidden": [32, 32], 
        "phi_hidden": [4, 4], #
        "eta_hidden": [4, 4], #
        "phi_output": 4,
        "eta_output": 4,
        "activation_function": nn.GELU(),
        "alpha":alpha,
        "beta": beta,
        "trainable_alpha": trainable_alpha,
        "a": a,
        "omega_z":omega_z,
        "omega_ho": omega_ho,
        "trainable_energy": trainable_energy,
        "beta_jastrow": beta_jastrow,
        "trainable_beta_jastrow":trainable_beta_jastrow,
    }