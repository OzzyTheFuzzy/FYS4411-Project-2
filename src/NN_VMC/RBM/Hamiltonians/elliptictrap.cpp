#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "elliptictrap.h"
#include "../particle.h"
#include "../WaveFunctions/wavefunction.h"

EllipticTrap::EllipticTrap(double gamma)
{
    assert(gamma > 0.0);
    m_gamma = gamma;
}

double EllipticTrap::computeLocalEnergy(
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
    if (!std::isfinite(laplacianPsi)) {
        return std::numeric_limits<double>::infinity();
    }

    const double kineticEnergy = -0.5 * (laplacianPsi / psi);

    double potentialEnergy = 0.0;
    for (auto& p : particles) {
        const auto& r = p->getPosition();

        if (r.empty()) {
            return std::numeric_limits<double>::infinity();
        }
        double r2 = 0.0;

        if (r.size() == 1) {
            // 1D harmonic oscillator
            r2 = r[0] * r[0];
        }
        else {
            // Generic elliptic trap:
            // D = 2: x^2 + gamma^2 y^2
            // D = 3: x^2 + y^2 + gamma^2 z^2
            // In general, gamma is applied to the last coordinate.
            for (std::size_t d = 0; d + 1 < r.size(); ++d) {
                r2 += r[d] * r[d];
            }

            const double last = r.back();
            r2 += m_gamma * m_gamma * last * last;
        }

        potentialEnergy += 0.5 * r2;
    }

    return kineticEnergy + potentialEnergy;
}