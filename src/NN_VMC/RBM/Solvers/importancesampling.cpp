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

    // Evaluate old wave function and old quantum force.
    // The quantum force is computed:
    //   - SimpleGaussian gives F = -4 alpha r
    //   - BoltzmannMachine gives the RBM force
    const double psiOld = waveFunction.evaluate(particles);
    const std::vector<double> F_old = waveFunction.computeQuantumForce(particles, i);

    // Propose new position using Langevin / Fokker-Planck step:
    //   R_new = R_old + D_diff * F_old * dt + sqrt(dt) * eta
    // where eta is a standard normal random vector.
    std::vector<double> newPos = oldPos;
    const double sqrtDt = std::sqrt(timeStep);
    for (unsigned int d = 0; d < D; d++) {
        const double gaussian = m_rng->nextGaussian(0.0, 1.0) * sqrtDt;
        newPos[d] += m_diffusion * timeStep * F_old[d] + gaussian;
    }

    // Move particle
    pos = newPos;

    const double psiNew = waveFunction.evaluate(particles);
    const std::vector<double> F_new = waveFunction.computeQuantumForce(particles, i);

    // Green's function ratio: G(old|new)/G(new|old)
    // G(y,x) ∝ exp[-(y - x - D_diff dt F(x))^2 / (4 D_diff dt)].
    double sumA2 = 0.0; // (old - new - D dt F_new)^2
    double sumB2 = 0.0; // (new - old - D dt F_old)^2
    for (unsigned int d = 0; d < D; d++) {
        const double A = (oldPos[d] - newPos[d] - m_diffusion * timeStep * F_new[d]);
        const double B = (newPos[d] - oldPos[d] - m_diffusion * timeStep * F_old[d]);
        sumA2 += A * A;
        sumB2 += B * B;
    }

    const double logGreensRatio = -(sumA2 - sumB2) / (4.0 * m_diffusion * timeStep);

    // Probability ratio:
    //   A = [G(old|new)/G(new|old)] * |Psi_new|^2 / |Psi_old|^2
    // Since the present wave functions are positive, we can use
    //   log A = logG + 2(log Psi_new - log Psi_old).
    if (psiOld <= 0.0 || psiNew <= 0.0) {
        pos = oldPos;
        return false;
    }

    const double logProbabilityRatio =
        logGreensRatio + 2.0 * (std::log(psiNew) - std::log(psiOld));

    if (logProbabilityRatio >= 0.0 ||
        std::log(m_rng->nextDouble()) < logProbabilityRatio) {
        return true; // accept
    } else {
        pos = oldPos; // reject
        return false;
    }
}