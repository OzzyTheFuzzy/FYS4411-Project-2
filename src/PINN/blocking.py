#statistical_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def blocking(E_array, B):
    """
    fimctopion that performs the blocking method on a given array of
    E: the array of values to be analyzed
    B: the size of the blocks
    """

    N=len(E_array)
    n = N//B              #number of blocks

    if n < 2:
        raise ValueError("Too few blocks to estimate variance.")

    E_trim = E_array[:n * B]    #trim the array to be a multiple of B
    blocks = E_trim.reshape(n, B)

    #calculate the block averages and then finding the mean of these averages
    block_means = np.mean(blocks, axis=1)
    mean = np.mean(block_means)

    # calculating the variance and the error of the mean
    variance = np.sum((block_means - mean) ** 2) / (n - 1)
    error = np.sqrt(variance / n)
    
    return variance, error, block_means


def blocking_error(E_array):
    """
    Function to calculate the standard error of the mean from the block averages
    E_array: array of the energies
    """
    variance_array, error_array = [], []
    idx=0
    N=len(E_array)
    n=0; k=0
    B_list=[]
    n_list=[]
    while True:  
        B = 2**k
        n = N//B 

        if n < 50:
            break
        B_list.append(B)
        n_list.append(n)
        
        variance, error, block_means = blocking(E_array, B)

        error_array.append(error)
        variance_array.append(variance)

        idx+=1
        k+=1
    
    return variance_array, error_array, B_list, n_list


def plot_blocking(x, y, beta, name_of_model, N, a):
    """ Plots the standard error of the mean as a function of block size B """

    name_of_plot=f"{name_of_model}_blocking_plot.pdf"

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Block size B")
    plt.ylabel("Standard error of the mean")
    plt.title(f"Standard error of the mean vs block size for N={N}, beta={beta}, a={a}")
    plt.grid(True)
    plt.savefig(f"figures/{name_of_plot}")
    plt.show()