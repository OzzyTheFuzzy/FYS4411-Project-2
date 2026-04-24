#include <cmath>
#include <vector>
#include <memory>
#include <cassert>

#include "importancesampling.h"
#include "WaveFunctions/wavefunction.h"
#include "particle.h"
#include "Math/random.h"

ImportanceSampling::ImportanceSampling(std::unique_ptr<class Random> rng, double diffusionConstant)
    : MonteCarlo(std::move(rng))
{
    assert(diffusionConstant > 0.0);
    m_diffusion = diffusionConstant;
}

bool ImportanceSampling::step(
    double timeStep,
    class WaveFunction& waveFunction,
    std::vector<std::unique_ptr<class Particle>>& particles
) {
    assert(timeStep > 0.0);

    const unsigned int N = particles.size();
    const unsigned int i = static_cast<unsigned int>(m_rng->nextDouble() * N);

    std::vector<double>& pos = particles[i]->getPosition();
    const unsigned int D = pos.size();

    const std::vector<double> oldPos = pos;

    // Evaluate Psi and get alpha (SimpleGaussian: parameters[0] = alpha)
    const double psiOld = waveFunction.evaluate(particles);
    const double alpha  = waveFunction.getParameters()[0];

    // Quantum force for SimpleGaussian: F = 2 ∇ ln Psi = -4α r
    std::vector<double> F_old(D, 0.0);
    for (unsigned int d = 0; d < D; d++) F_old[d] = -4.0 * alpha * oldPos[d];

    // Propose new position with Langevin step:
    // y = x + D*F(x)*dt + ξ*sqrt(dt),  ξ ~ N(0,1)
    std::vector<double> newPos = oldPos;
    const double sqrtDt = std::sqrt(timeStep);
    for (unsigned int d = 0; d < D; d++) {
        const double gaussian = m_rng->nextGaussian(0.0, 1.0) * sqrtDt;
        newPos[d] += m_diffusion * timeStep * F_old[d] + gaussian;
    }

    // Move particle
    pos = newPos;

    const double psiNew = waveFunction.evaluate(particles);

    // Quantum force at new position
    std::vector<double> F_new(D, 0.0);
    for (unsigned int d = 0; d < D; d++) F_new[d] = -4.0 * alpha * newPos[d];

    // Green's function ratio: G(old|new)/G(new|old)
    // G(y,x) ∝ exp(-(y-x-D dt F(x))^2/(4 D dt))
    double sumA2 = 0.0; // (old - new - D dt F_new)^2
    double sumB2 = 0.0; // (new - old - D dt F_old)^2
    for (unsigned int d = 0; d < D; d++) {
        const double A = (oldPos[d] - newPos[d] - m_diffusion * timeStep * F_new[d]);
        const double B = (newPos[d] - oldPos[d] - m_diffusion * timeStep * F_old[d]);
        sumA2 += A * A;
        sumB2 += B * B;
    }

    const double logG = -(sumA2 - sumB2) / (4.0 * m_diffusion * timeStep);
    const double greensRatio = std::exp(logG);

    const double probRatio = greensRatio * (psiNew * psiNew) / (psiOld * psiOld);

    if (probRatio >= 1.0 || m_rng->nextDouble() < probRatio) {
        return true; // accept
    } else {
        pos = oldPos; // reject
        return false;
    }
}