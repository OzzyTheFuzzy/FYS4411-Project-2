# FYS4411 Project 2

This repository contains the code developed for Project 2 in FYS4411. The project studies bosonic systems in harmonic traps using neural-network-based approaches to represent or approximate the many-body wave function.

The main code is located in the `src/` directory. Each method is implemented in a separate subfolder and contains its own README with more detailed instructions.

## Repository structure

```text
FYS4411-Project-2/
├── src/
│   ├── NN_VMC/
│   ├── ffwd_NN_VMC/
│   └── PINN/
└── README.md
```

## `src/NN_VMC`

This folder contains the Variational Monte Carlo (VMC) implementation based on a Restricted Boltzmann Machine (RBM) trial wave function.

The code builds on the VMC structure from Project 1, replacing the analytical trial wave function by a Gaussian-binary RBM. 

Numerical results, plots, and post-processing scripts are stored inside its `data/` folder.

## `src/ffwd_NN_VMC`

This folder contains the VMC implementation using a feedforward neural network as the variational wave function.

The method follows the neural-network ansatz inspired by Saito's approach. 

This implementation is used to compare the performance of a more flexible feedforward neural network against the RBM ansatz.

General logs, numerical results, plots are stored inside folders `logs/`, `logs_NN/`, `logs_OBD/`, `figs_NN/`.
More details on how these folders work are explained in `./src/fffwd_NN_VMC/README.md`.

## `src/PINN`

This folder contains the Physics-Informed Neural Network implementation.

The PINN approach does not use standard VMC optimization of a trial wave function. Instead, it trains a neural network by minimizing a residual of the Schrödinger equation. The implementation includes sampling of physically relevant particle configurations, treatment of isotropic and anisotropic traps, and interaction terms. Blocking method plots, which where used for estimation of the uncertainty of the energy can be found in PINN/figures. All energy results, including the variance, can be found in PINN/final_energy_eval and all the training data, model structures and training parameters can be found in PINN/logs.

## Notes

Each method folder contains its own README with instructions for building, running, and analyzing the corresponding code. The top-level `src/` directory is organized so that the three approaches can be developed and tested independently while addressing the same physical systems and benchmarks.
