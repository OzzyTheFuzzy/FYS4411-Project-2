#pragma once

#include <vector>
#include <memory>
#include "wavefunction.h"

/**
 * @brief Optimization class caching logarithmic wave function values particle by particle 
 */
class WaveFunctionCache {
public:
    WaveFunctionCache(WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles);

    /**
     * @brief Computes the ratio ln(Psi_new) - ln(Psi_old) for a proposed move.
     * @param particles The system particles (with the target particle in its proposed new position).
     * @param particle_idx The index of the particle that was moved.
     * @return The logarithmic difference in wave function probability.
     */
    double computeLnRatio(std::vector<std::unique_ptr<class Particle>>& particles,
        unsigned int particle_idx);

    /**
     * @brief Finalizes the cache state if the proposed move was accepted by Metropolis.
     */
    void acceptMove(std::vector<std::unique_ptr<class Particle>>& particles);

    double getParticleLn(unsigned int i) const { return m_particleLn[i]; }

    /**
     * @brief Computes the Laplacian numerically, utilizing cached values to minimize redundant math.
     */
    double computeNumericalLaplacian(
        std::vector<std::unique_ptr<Particle>>& particles,
        WaveFunction& waveFunction);

private:
    WaveFunction& m_wf;
    std::vector<double> m_particleLn;   ///< Stores the ln contribution of each particle
    double m_totalLn = 0.0;             ///< The total system ln(|Psi|)
    double m_pendingLn = 0.0;           ///< The pending value of the particle currently being tested
    unsigned int m_particleToUpd;
};