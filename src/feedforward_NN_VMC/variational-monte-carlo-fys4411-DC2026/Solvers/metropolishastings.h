#pragma once

#include <memory>

#include "montecarlo.h"
#include "../WaveFunctions/wavefunctioncache.h"

/**
 * @brief Implements the Metropolis-Hastings algorithm with Importance Sampling.
 * * Uses the Langevin equation to guide particles towards regions of higher 
 * probability density, guided by the "quantum force" (gradient of the wave function).
 * This significantly improves the acceptance ratio and convergence speed.
 */
class MetropolisHastings : public MonteCarlo {
public:
    MetropolisHastings(std::unique_ptr<class Random> rng, bool preferAnalytic = true, bool useCache = true);
    bool step(double timeStep, class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles);
    bool get_preferAnalytic() override { return m_preferAnalytic; };
    bool hasAnalyticalOption() override { return true; }
private:
    const double m_D = 0.5;
};
