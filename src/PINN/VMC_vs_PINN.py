import numpy as np
import matplotlib.pyplot as plt



def compute_rel_and_abs_error(C_numerical, C_model, epsilon=1e-4):
    """
    Computes the relative and absolute error between the numerical solution and the model's predicted values.

    Parameters:
        - C_numerical  (np.ndarray): Numerical solution to the Black-Scholes equation.
        - C_model      (np.ndarray): Predicted option prices from the model
        - epsilon (float, optional): Small value to prevent division by zero in relative error

    Returns:
        - rel_error (np.ndarray): Relative error array, same shape as inputs.
        - abs_error (np.ndarray): Absolute error array, same shape as inputs.
    """
    abs_error = np.abs(C_model - C_numerical)
    rel_error = abs_error / (np.abs(C_numerical) + epsilon)

    return rel_error, abs_error



def plot_rel_error_scatter(S_total, t, rel_error_flat, title):
    """
    Creates a scatter plot of the relative error over total asset price and time.

    Parameters:
        - S_total        (np.ndarray): Total asset prices (S1 + S2), flattened array.
        - t              (np.ndarray): Time values corresponding to each S_total value.
        - rel_error_flat (np.ndarray): Flattened array of relative error values.
        - title                 (str): Plot title.
    """
    # Determine color scale limits from the error values
    vmin, vmax = np.min(rel_error_flat), np.max(rel_error_flat)

    # Create scatter plot
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        S_total, t, c=rel_error_flat, cmap="viridis",
        s=50, edgecolor="k", vmin=vmin, vmax=vmax
    )
    plt.colorbar(scatter, label="Relative Error")
    plt.title(f"Relative Error: {title}")
    plt.xlabel(r"$S_{\text{tot}}$")
    plt.ylabel(r"$t$ [years]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()