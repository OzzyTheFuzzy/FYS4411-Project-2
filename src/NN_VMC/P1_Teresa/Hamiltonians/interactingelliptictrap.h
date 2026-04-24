#pragma once
#include <memory>
#include <vector>

#include "hamiltonian.h"

class InteractingEllipticTrap : public Hamiltonian {
public:
    InteractingEllipticTrap(double gamma);

    double computeLocalEnergy(
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles
    ) override;

private:
    double m_gamma = 1.0;
};