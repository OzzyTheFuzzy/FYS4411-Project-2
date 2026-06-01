#pragma once

#include <memory>
#include <vector>

#include "hamiltonian.h"

class RBMInteractingTrap : public Hamiltonian {
public:
    RBMInteractingTrap(
        double gamma = 1.0,
        double interactionStrength = 1.0,
        double epsilon = 1e-12
    );

    double computeLocalEnergy(
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles
    ) override;

private:
    double m_gamma = 1.0;
    double m_interactionStrength = 1.0;
    double m_epsilon = 1e-12;

    double computeTrapPotential(const std::vector<double>& r) const;
    double computePairDistance(
        const std::vector<double>& ri,
        const std::vector<double>& rj
    ) const;
};