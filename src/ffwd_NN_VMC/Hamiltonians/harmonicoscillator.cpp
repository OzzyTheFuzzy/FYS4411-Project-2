#include <memory>
#include <cassert>
#include <iostream>

#include "../common.h"
#include "harmonicoscillator.h"
#include "../particle.h"
#include "../WaveFunctions/wavefunction.h"
#include "../WaveFunctions/wavefunctioncache.h"

using namespace CommonUtils;

HarmonicOscillator::HarmonicOscillator(double omega) {
    if (omega <= 0) throw std::invalid_argument("omega needs to be a positive value");
    m_omega = omega;
}

double HarmonicOscillator::computeLocalEnergy(
    class WaveFunction& waveFunction,
    std::vector<std::unique_ptr<class Particle>>& particles,
    WaveFunctionCache& cache
) {
    /* Here, you need to compute the kinetic and potential energies.
     * Access to the wave function methods can be done using the dot notation
     * for references, e.g., wavefunction.computeDoubleDerivative(particles),
     * to get the Laplacian of the wave function.
     * */

    double kineticEnergy;
    if (waveFunction.hasAnalyticalDerivative() && m_analytic_ifAvailable) {
        kineticEnergy = -0.5 * waveFunction.computeDoubleDerivative(particles) / waveFunction.evaluate(particles);
    }
    else {
        // the following commented line is deprecated
        // kineticEnergy = -0.5 * waveFunction.computeNumericalDoubleDerivative(particles) / waveFunction.evaluate(particles);
        kineticEnergy = -0.5 * cache.computeNumericalLaplacian(particles, waveFunction);
    }

    double sum = 0;
    for (unsigned int i = 0; i < particles.size(); i++) {
        for (unsigned int j = 0; j < particles[0]->getNumberOfDimensions(); j++) {
            sum += sq(particles[i]->getPosition()[j]);
        }
    }
    double potentialEnergy = 0.5 * sq(m_omega) * sum; // * m

    return kineticEnergy + potentialEnergy;
}
