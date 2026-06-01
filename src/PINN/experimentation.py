from PINN_vs_analytical import *
from blocking import blocking_error, plot_blocking
import config_experimentation as config # import configuration parameters for experimentation
from train_and_save import train_and_save

model_name   = f"{config.N}N_d{config.dim}_beta{config.beta}_a{config.a}_width{config.width}_IG{config.initialize_gaussian}" # name for saving model and logs

#model_name= f"{config.N}N_d{config.dim}_beta{config.beta}_a{config.a}_width{config.width}_Etunable_IG{config.initialize_gaussian}" # uncomment and write your specific model name for loading a trained model and plotting results

#train_and_save(config, model_name, config.model_config) # uncomment to train and save training data for the given config 
#plot_loss_curves(model_name, config) # for plotting the loss during training

# vmc samples from .dat file
if config.a==0.0:
    samples = 800000

else:
    samples = 524288

full = True # set True for eval over many samples. Set to False for only 10,000 points
E_mean, E_std, E_L= energy_vmc_and_plot(model_name,config, full=full) # for evaluating the energy of the trained model

block_variance, block_error, B_list, n_list = blocking_error(E_L.flatten()) # for performing blocking analysis on the energies obtained from the trained model

plot_blocking(B_list, block_error, beta=config.beta, name_of_model=model_name, N=config.N, a=config.a) # for plotting the blocking analysis results