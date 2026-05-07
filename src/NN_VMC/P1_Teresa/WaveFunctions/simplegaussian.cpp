#include <memory>
#include <cmath>
#include <cassert>

#include "simplegaussian.h"
#include "wavefunction.h"
#include "../system.h"
#include "../particle.h"

SimpleGaussian::SimpleGaussian(double alpha)
{
    assert(alpha >= 0);
    m_numberOfParameters = 1;
    m_parameters.reserve(1);
    m_parameters.push_back(alpha);
}

double SimpleGaussian::evaluate(std::vector<std::unique_ptr<class Particle>>& particles) {
    // General Gaussian trial wave function for arbitrary N and D:
    // Psi(R) = exp(-alpha * sum_{i,d} x_{id}^2)

    const double alpha = m_parameters[0];

    double sum_r2 = 0.0;
    for (auto& p : particles) {
        const std::vector<double>& r = p->getPosition();
        for (double x : r) {
            sum_r2 += x * x;
        }
    }
    return std::exp(-alpha * sum_r2);
}

double SimpleGaussian::computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles) {
    const double alpha = m_parameters[0];

    const int N = static_cast<int>(particles.size());
    const int D = static_cast<int>(particles[0]->getPosition().size());

    // Sum over i of r_i^2
    double sum_r2 = 0.0;
    for (int i = 0; i < N; i++) {
        const std::vector<double>& r = particles[i]->getPosition();
        for (int d = 0; d < D; d++) {
            sum_r2 += r[d] * r[d];
        }
    }

    const double psi = evaluate(particles);

    // ∑_i ∇_i^2 Ψ = Ψ [ -2α D N + 4 α^2 ∑_i r_i^2 ]
    return psi * (-2.0 * alpha * D * N + 4.0 * alpha * alpha * sum_r2);
}

std::vector<double> SimpleGaussian::computeQuantumForce(std::vector<std::unique_ptr<class Particle>>& particles, unsigned int particleIndex) {
    const double alpha = m_parameters[0];

    const std::vector<double>& r = particles[particleIndex]->getPosition();
    std::vector<double> force(r.size(), 0.0);

    // For Psi = exp(-alpha * sum r_i^2):
    //
    //   grad_i ln Psi = -2 alpha r_i
    //   F_i = 2 grad_i ln Psi = -4 alpha r_i
    for (unsigned int d = 0; d < r.size(); d++) {
        force[d] = -4.0 * alpha * r[d];
    }

    return force;
}

double SimpleGaussian::computeLogDerivativeAlpha(std::vector<std::unique_ptr<class Particle>>& particles)
{
    // O_alpha(R) = d/dalpha ln Psi_alpha(R)
    // For Psi = exp(-alpha * sum r_i^2), this gives:
    // O_alpha(R) = - sum r_i^2

    double sum_r2 = 0.0;
    for (auto& p : particles){
        const std::vector<double>& r = p->getPosition();
        for (double x : r){
            sum_r2 += x * x;
        }
    }
    return -sum_r2;
}
