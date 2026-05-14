#pragma once
#include <memory>
#include <vector>

#include "hamiltonian.h"

/**
 * @brief Represents an ideal, non-interacting Bose gas in a spherical harmonic trap.
 * * This Hamiltonian models the standard quantum harmonic oscillator.
 */
class HarmonicOscillator : public Hamiltonian {
public:
    /**
     * @brief Constructs the ideal harmonic oscillator.
     * @param omega The trap frequency (strength of the confinement).
     */
    HarmonicOscillator(double omega);
    double computeLocalEnergy(
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles,
        class WaveFunctionCache& cache
    );

private:
    double m_omega;
};

