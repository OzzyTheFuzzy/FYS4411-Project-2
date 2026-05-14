#pragma once

#include <memory>

#include "montecarlo.h"
#include "../WaveFunctions/wavefunctioncache.h"

/**
 * @brief Implements the standard Metropolis algorithm (Brute Force).
 * * Proposes symmetric, uniformly distributed random moves for the particles.
 */
class Metropolis : public MonteCarlo {
public:
    Metropolis(std::unique_ptr<class Random> rng, bool preferAnalytic = true, bool useCache = true);
    bool step(
        double stepLength,
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles);
    bool get_preferAnalytic() override { return m_preferAnalytic; };
    bool hasAnalyticalOption() override { return true; }
};
