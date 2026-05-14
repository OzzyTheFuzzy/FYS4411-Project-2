#pragma once

#include <memory>

#include "wavefunction.h"

/**
 * @brief Represents a simple, isotropic Gaussian wave function.
 * * Used for non-interacting bosons in a perfectly spherical trap.
 * Contains a single variational parameter, alpha.
 * Equation: Psi = exp(-alpha * sum(r_i^2)).
 */
class SimpleGaussian : public WaveFunction {
public:
    SimpleGaussian(double alpha);
    double evaluate(std::vector<std::unique_ptr<class Particle>>& particles);
    double computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles);
    double computeParticleLn(std::vector<std::unique_ptr<class Particle>>& particles,
        unsigned int particle_idx);
    bool hasAnalyticalDerivative() override { return true; }

    std::vector<double> lowerBounds() const override { return { 1e-3 }; }
    std::vector<double> upperBounds() const override { return { 1.0 }; }
};
 