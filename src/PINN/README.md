# PINN

This folder contains the Physics-Informed Neural Network (PINN) implementation used to compute ground-state energies of the harmonic oscillator trap, with and without interactions.

## Main Files

### `experimentation.py`

Main script for training and evaluating models.

Physical parameters:

* `N`: number of particles
* `dim`: spatial dimension
* `a`: interaction strength / hard-core radius
* `beta`: anisotropy parameter 

Training parameters:

* `training_points`
* `epochs`
* `batch_size`
* `lr`

To train a model, uncomment:

```python
train_and_evaluate()
```

and run:

```bash
python experimentation.py
```

The script:

1. Generates collocation points.
2. Builds the PINN model.
3. Trains the network.
4. Saves the model in `models/`.
5. Saves training logs in `logs/`.
6. Evaluates the trained model.
7. Performs blocking analysis, create and saves plots of the blocking analysis.

### `initialize_data.py`

Generates the training and validation collocation points used by the PINN.
It supports:

* non-interacting systems (`a = 0`), where positions are sampled from Gaussian distributions.
* interacting systems, where proposals are sampled from diffrerent gaussians and where proposals are filtered with a minimum-distance constraint,
* an alternative ellipsoidal sampler (`particle_pos2`) for the anisotropic traps.


### `model.py`

Defines the neural-network wavefunction (`SE_Model`).

### `training.py`

Contains the training loop, validation routines, and optimization procedure.

### `loss_function.py`

Computes the Schrödinger-equation residual and local-energy quantities used for training and validation of training.

### `PINN_vs_analytical.py`

Evaluates trained PINN models and compares results with analytical solutions.

### `VMC_vs_PINN.py`

Compares PINN results with VMC calculations. Also has helper functions that:
* reads .pth files and reconstructs models
* load results from the -json files
* plots losses from training
* plots energies for the VMC sampled configuration

### `blocking.py`

Performs blocking analysis and uncertainty estimation.

### `load_positions.py`

Loads saved particle-positions from configurations from VMC simulations

---

## Folders

### `models/`

Saved trained models (`.pth` files).

### `logs/`

Training logs and metadata (`.json` files).

### `figures/`

Generated figures.

### `positions/`

Stored particle-position samples.

### `__pycache__/`

Python cache files (can be ignored).

---

## Creating Data

Training data is generated automatically through `initialize_data.py` when running `experimentation.py`.

Important parameters:

```python
training_point:
width 
seed 
```

To generate data for a different system, modify parameters such as:

```python
N = 2
dim = 3
a = 0.0
beta = 1.0
```

and rerun:

```bash
python experimentation.py
```