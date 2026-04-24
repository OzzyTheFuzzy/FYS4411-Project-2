#include<memory>
#include <cassert>
#include <iostream>

#include "harmonicoscillator.h"
#include "../particle.h"
#include "../WaveFunctions/wavefunction.h"
#include "../WaveFunctions/simplegaussian.h"
#include <cmath>

using std::cout;
using std::endl;

HarmonicOscillator::HarmonicOscillator(double omega, bool useNumericalLaplacian, double h)
{
    assert(omega > 0);
    assert(h > 0);
    m_omega  = omega;
    m_useNumericalLaplacian = useNumericalLaplacian;
    m_h = h;
}

static double numericalLaplacian(
    class WaveFunction& waveFunction,
    std::vector<std::unique_ptr<class Particle>>& particles,
    double h
) {
    const double h2 = h * h;

    // Try optimized path for SimpleGaussian
    SimpleGaussian* gaussian = dynamic_cast<SimpleGaussian*>(&waveFunction);

    if (gaussian != nullptr) {
        const double alpha = gaussian->getParameters()[0];

        const unsigned int N = particles.size();
        const unsigned int D = particles[0]->getNumberOfDimensions();

        // Compute S = sum_{i,d} x_{id}^2 once
        double S = 0.0;
        for (unsigned int i = 0; i < N; i++) {
            const std::vector<double>& pos = particles[i]->getPosition();
            for (unsigned int d = 0; d < D; d++) {
                S += pos[d] * pos[d];
            }
        }

        const double psi0 = std::exp(-alpha * S);
        double lap = 0.0;

        for (unsigned int i = 0; i < N; i++) {
            const std::vector<double>& pos = particles[i]->getPosition();

            for (unsigned int d = 0; d < D; d++) {
                const double x = pos[d];

                const double psiPlus  = psi0 * std::exp(-alpha * ( 2.0 * x * h + h2));
                const double psiMinus = psi0 * std::exp(-alpha * (-2.0 * x * h + h2));

                lap += (psiPlus - 2.0 * psi0 + psiMinus) / h2;
            }
        }

        return lap; // approximates ∇²Ψ
    }

    // Generic fallback: full finite-difference reevaluation
    const double psi0 = waveFunction.evaluate(particles);
    double lap = 0.0;

    const unsigned int N = particles.size();
    const unsigned int D = particles[0]->getNumberOfDimensions();

    for (unsigned int i = 0; i < N; i++) {
        std::vector<double>& pos = particles[i]->getPosition();

        for (unsigned int d = 0; d < D; d++) {
            const double x_old = pos[d];

            pos[d] = x_old + h;
            const double psiPlus = waveFunction.evaluate(particles);

            pos[d] = x_old - h;
            const double psiMinus = waveFunction.evaluate(particles);

            pos[d] = x_old;

            lap += (psiPlus - 2.0 * psi0 + psiMinus) / h2;
        }
    }

    return lap;
}

double HarmonicOscillator::computeLocalEnergy(
            class WaveFunction& waveFunction,
            std::vector<std::unique_ptr<class Particle>>& particles
        )
{
    /* Here, you need to compute the kinetic and potential energies.
     * Access to the wave function methods can be done using the dot notation
     * for references, e.g., wavefunction.computeDoubleDerivative(particles),
     * to get the Laplacian of the wave function.
     * */

    // Natural units: hbar = m = 1
    // Local energy: E_L = -1/2 * (∇^2Ψ / Ψ) + 1/2 * ω^2 * Σ_i r_i^2

    const double psi = waveFunction.evaluate(particles);

    double laplacianPsi = 0.0;
    if (m_useNumericalLaplacian) {
        laplacianPsi = numericalLaplacian(waveFunction, particles, m_h);
    } else {
        laplacianPsi = waveFunction.computeDoubleDerivative(particles);
    }

    // Kinetic energy
    const double kineticEnergy = -0.5 * (laplacianPsi / psi);

    // Potential energy
    double r2Sum = 0.0;
    for (auto& p : particles) {
        const std::vector<double>& r = p->getPosition();
        for (double x : r) r2Sum += x * x;
    }
    const double potentialEnergy = 0.5 * m_omega * m_omega * r2Sum;

    return kineticEnergy + potentialEnergy;
}
