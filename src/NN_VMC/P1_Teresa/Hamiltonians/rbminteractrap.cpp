#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "rbminteractrap.h"
#include "../particle.h"
#include "../WaveFunctions/wavefunction.h"

RBMInteractingTrap::RBMInteractingTrap(
    double omega,
    double gamma,
    double interactionStrength,
    double epsilon
)
{
    assert(omega > 0.0);
    assert(gamma > 0.0);
    assert(epsilon > 0.0);

    m_omega = omega;
    m_gamma = gamma;
    m_interactionStrength = interactionStrength;
    m_epsilon = epsilon;
}

double RBMInteractingTrap::computeLocalEnergy(
    class WaveFunction& waveFunction,
    std::vector<std::unique_ptr<class Particle>>& particles
)
{
    const double psi = waveFunction.evaluate(particles);

    // If the wave function is zero or numerically invalid, reject this configuration
    // by assigning an infinite local energy.
    if (psi == 0.0 || !std::isfinite(psi)) {
        return std::numeric_limits<double>::infinity();
    }

    const double laplacianPsi = waveFunction.computeDoubleDerivative(particles);
    const double kineticEnergy = -0.5 * (laplacianPsi / psi);

    // Harmonic confinement:
    // For D = 2:
    //   V_trap = 1/2 * omega^2 * (x^2 + gamma^2 y^2)
    // For D = 3:
    //   V_trap = 1/2 * omega^2 * (x^2 + y^2 + gamma^2 z^2)
    double trapPotential = 0.0;
    for (auto& p : particles) {
        const auto& r = p->getPosition();
        trapPotential += computeTrapPotential(r);
    }

    // Coulomb-like repulsion:
    //   V_int = sum_{i<j} interactionStrength / r_ij
    double interactionPotential = 0.0;

    for (std::size_t i = 0; i < particles.size(); ++i) {
        const auto& ri = particles[i]->getPosition();

        for (std::size_t j = i + 1; j < particles.size(); ++j) {
            const auto& rj = particles[j]->getPosition();

            const double rij = computePairDistance(ri, rj);

            // Avoid numerical singularity if two particles overlap exactly.
            if (rij < m_epsilon) {
                return std::numeric_limits<double>::infinity();
            }

            interactionPotential += m_interactionStrength / rij;
        }
    }

    return kineticEnergy + trapPotential + interactionPotential;
}

double RBMInteractingTrap::computeTrapPotential(
    const std::vector<double>& r
) const
{
    const std::size_t D = r.size();

    double r2 = 0.0;

    if (D == 1) {
        r2 = r[0] * r[0];
    }
    else {
        // Gamma is applied to the last coordinate.
        //
        // D = 2: x^2 + gamma^2 y^2
        // D = 3: x^2 + y^2 + gamma^2 z^2
        for (std::size_t d = 0; d + 1 < D; ++d) {
            r2 += r[d] * r[d];
        }

        r2 += m_gamma * m_gamma * r[D - 1] * r[D - 1];
    }

    return 0.5 * m_omega * m_omega * r2;
}

double RBMInteractingTrap::computePairDistance(
    const std::vector<double>& ri,
    const std::vector<double>& rj
) const
{
    assert(ri.size() == rj.size());

    double distanceSquared = 0.0;

    for (std::size_t d = 0; d < ri.size(); ++d) {
        const double diff = ri[d] - rj[d];
        distanceSquared += diff * diff;
    }

    return std::sqrt(distanceSquared);
}