#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "interactingelliptictrap.h"
#include "../particle.h"
#include "../WaveFunctions/wavefunction.h"

InteractingEllipticTrap::InteractingEllipticTrap(double gamma)
{
    assert(gamma > 0.0);
    m_gamma = gamma;
}

double InteractingEllipticTrap::computeLocalEnergy(
    class WaveFunction& waveFunction,
    std::vector<std::unique_ptr<class Particle>>& particles
)
{
    const double psi = waveFunction.evaluate(particles);

    // Forbidden hard-core configuration
    if (psi == 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    const double laplacianPsi = waveFunction.computeDoubleDerivative(particles);
    const double kineticEnergy = -0.5 * (laplacianPsi / psi);

    double potentialEnergy = 0.0;
    for (auto& p : particles) {
        const auto& r = p->getPosition();
        potentialEnergy += 0.5 * (r[0]*r[0] + r[1]*r[1] + m_gamma*m_gamma*r[2]*r[2]);
    }

    return kineticEnergy + potentialEnergy;
}