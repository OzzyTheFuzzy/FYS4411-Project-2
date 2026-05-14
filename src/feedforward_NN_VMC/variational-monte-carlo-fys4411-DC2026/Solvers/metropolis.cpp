#include <memory>
#include <vector>

#include "common.h"
#include "metropolis.h"
#include "WaveFunctions/wavefunction.h"
#include "particle.h"
#include "Math/random.h"

using namespace CommonUtils;

Metropolis::Metropolis(std::unique_ptr<class Random> rng, bool preferAnalytic, bool useCache)
    : MonteCarlo(std::move(rng), preferAnalytic, useCache) {}


bool Metropolis::step(
    double stepLength,
    class WaveFunction& waveFunction,
    std::vector<std::unique_ptr<class Particle>>& particles) {
    /* Perform the actual Metropolis step: Choose a particle at random and
     * change its position by a random amount, and check if the step is
     * accepted by the Metropolis test (compare the wave function evaluated at
     * this new position with the one at the old position).
     */
    if (m_useCache && !m_cache)
        m_cache = std::make_unique<WaveFunctionCache>(waveFunction, particles);

    double psi_old = m_useCache ? 0 : waveFunction.evaluate(particles);

    unsigned int particle_idx = m_rng->nextInt(particles.size() - 1);

    unsigned int numberOfDimensions = particles[particle_idx]->getNumberOfDimensions();
    std::vector<double> displacement(numberOfDimensions);
    for (unsigned int i = 0; i < numberOfDimensions; i++) {
        displacement[i] = (m_rng->nextDouble() - .5) * stepLength;
        particles[particle_idx]->adjustPosition(displacement[i], i);
    }

    double ratio = m_useCache ?
        m_cache->computeLnRatio(particles, particle_idx) : waveFunction.evaluate(particles) / psi_old;

    bool accepted = m_useCache ?
        m_rng->nextDouble() <= exp(2.0 * ratio) : m_rng->nextDouble() <= sq(ratio);
    if (accepted) {
        if (m_useCache) {
            m_cache->acceptMove(particles);
        }
    }
    else {
        for (unsigned int i = 0; i < numberOfDimensions; i++) {
            particles[particle_idx]->adjustPosition(-displacement[i], i);
        }
    }

    return accepted;
}
