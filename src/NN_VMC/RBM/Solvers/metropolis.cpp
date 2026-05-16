#include <memory>
#include <vector>

#include "metropolis.h"
#include "WaveFunctions/wavefunction.h"
#include "particle.h"
#include "Math/random.h"


Metropolis::Metropolis(std::unique_ptr<class Random> rng)
    : MonteCarlo(std::move(rng))
{
}


bool Metropolis::step(
        double stepLength,
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles)
{
    /* Perform the actual Metropolis step: Choose a particle at random and
     * change its position by a random amount, and check if the step is
     * accepted by the Metropolis test (compare the wave function evaluated at
     * this new position with the one at the old position).
     */

    const unsigned int N = static_cast<unsigned int>(particles.size());
    const unsigned int particleIndex =
        static_cast<unsigned int>(m_rng->nextDouble() * N);

    // Get modifiable reference to chosen particle position
    std::vector<double>& pos = particles[particleIndex]->getPosition();
    const unsigned int D = static_cast<unsigned int>(pos.size());

    // Store old position so we can revert on rejection
    const std::vector<double> oldPos = pos;

    // Evaluate wave function at current configuration
    const double psiOld = waveFunction.evaluate(particles);

    // Propose new position: uniform displacement in each dimension
    for (unsigned int d = 0; d < D; d++) {
        const double delta = (m_rng->nextDouble() * 2.0 - 1.0) * stepLength;
        pos[d] += delta;
        // alternatively: particles[particleIndex]->adjustPosition(delta, d);
    }

    // Evaluate wave function at proposed configuration
    const double psiNew = waveFunction.evaluate(particles);

    // Metropolis acceptance ratio: |Psi_new|^2 / |Psi_old|^2
    const double ratio = (psiNew * psiNew) / (psiOld * psiOld);

    // Accept/reject
    if (ratio >= 1.0 || m_rng->nextDouble() < ratio) {
        return true;   // accept: keep new position
    } else {
        pos = oldPos;  // reject: revert
        return false;
    }
}
