#pragma once

#include <memory>
#include <vector>

#include "../particle.h"
#include "Math/random.h"

/**
 * @brief Sets up the initial random configuration of particles in the system.
 * * This function generates a starting state for the Variational Monte Carlo simulation.
 * It places particles randomly within a defined bounding box using a uniform distribution.
 * If a repulsive hard-core diameter (rep_a) is provided, it enforces that no two particles 
 * are initialized closer than this distance (rejection sampling).
 *
 * @param numberOfDimensions The spatial dimensions of the system (e.g., 1, 2, or 3).
 * @param numberOfParticles The total number of particles to place in the trap.
 * @param randomEngine Reference to the random number generator used for placement.
 * @param rep_a The hard-core repulsion diameter. Defaults to 0 (ideal gas).
 * @return std::vector<std::unique_ptr<Particle>> A vector containing the initialized particles.
 */
std::vector<std::unique_ptr<Particle>> setupRandomUniformInitialState(
    unsigned int numberOfDimensions,
    unsigned int numberOfParticles,
    Random& randomEngine,
    double rep_a = 0
);
