#pragma once
#include <memory>
#include <vector>

#include "hamiltonian.h"

class HarmonicOscillator : public Hamiltonian {
public:
    HarmonicOscillator(double omega, bool useNumericalLaplacian=false, double h=1e-3);
    double computeLocalEnergy(
            class WaveFunction& waveFunction,
            std::vector<std::unique_ptr<class Particle>>& particles) override;

private:
    double m_omega = 1.0;
    bool m_useNumericalLaplacian = false;
    double m_h = 1e-3;
};

