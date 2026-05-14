#pragma once

#include <memory>

#include "wavefunction.h"

/**
 * @brief Represents an interacting (repulsive) wave function in an elliptical trap.
 * * Uses a Gaussian single-particle state combined with a Jastrow correlation factor 
 * to model the hard-core repulsion between bosons.
 * Parameters: alpha (xy width), beta (z deformation), and rep_a (hard-core radius).
 */
class BoltzmannMachine : public WaveFunction {
public:
    BoltzmannMachine(double alpha, double beta, double rep_a);
    double evaluate(std::vector<std::unique_ptr<class Particle>>& particles);
    double evaluateLn(std::vector<std::unique_ptr<class Particle>>& particles);

    /**
     * @brief Evaluates only the non-interacting Gaussian part in log space.
     */
    double evaluateLn_noInteraction(std::vector<std::unique_ptr<class Particle>>& particles);
    double computeParticleLn(std::vector<std::unique_ptr<Particle>>& particles,
        unsigned int particle_idx);

    // Due to the complexity of the Jastrow factor, we use the WaveFunctionCache 
    // for numerical differentiation of the Laplacian.
    bool hasAnalyticalDerivative() override { return false; }

    std::vector<double> lowerBounds() const override { return { 1e-3, 0.1 }; }
    std::vector<double> upperBounds() const override { return { 1.0, 5.0 }; }

private:
    const unsigned int m_NDIM = 3;
    double m_rep_a;
};
