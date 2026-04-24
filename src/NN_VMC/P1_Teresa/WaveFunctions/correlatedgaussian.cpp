#include <cmath>
#include <cassert>
#include <memory>
#include <vector>

#include "correlatedgaussian.h"
#include "../particle.h"

CorrelatedGaussian::CorrelatedGaussian(double alpha, double beta, double a)
{
    assert(alpha >= 0.0);
    assert(beta  >  0.0);
    assert(a     >= 0.0);

    m_numberOfParameters = 3;
    m_parameters = {alpha, beta, a};
}

double CorrelatedGaussian::pairDistance(
    const std::vector<double>& ri,
    const std::vector<double>& rj
) const
{
    double r2 = 0.0;
    for (unsigned int d = 0; d < ri.size(); ++d) {
        const double diff = ri[d] - rj[d];
        r2 += diff * diff;
    }
    return std::sqrt(r2);
}

double CorrelatedGaussian::evaluate(std::vector<std::unique_ptr<class Particle>>& particles)
{
    const double alpha = m_parameters[0];
    const double beta  = m_parameters[1];
    const double a     = m_parameters[2];

    const unsigned int N = static_cast<unsigned int>(particles.size());
    const unsigned int D = particles[0]->getNumberOfDimensions();
    assert(D == 3);

    // One-body Gaussian part
    double exponent = 0.0;
    for (unsigned int i = 0; i < N; ++i) {
        const auto& r = particles[i]->getPosition();
        exponent += r[0]*r[0] + r[1]*r[1] + beta*r[2]*r[2];
    }

    double psi = std::exp(-alpha * exponent);

    // Jastrow pair factor
    for (unsigned int i = 0; i < N; ++i) {
        const auto& ri = particles[i]->getPosition();
        for (unsigned int j = i + 1; j < N; ++j) {
            const auto& rj = particles[j]->getPosition();
            const double rij = pairDistance(ri, rj);

            if (rij <= a) {
                return 0.0; // forbidden hard-core overlap
            }

            psi *= (1.0 - a / rij);
        }
    }

    return psi;
}

double CorrelatedGaussian::computeLogDerivativeAlpha(
    std::vector<std::unique_ptr<class Particle>>& particles)
{
    // O_alpha(R) = d/dalpha ln Psi = - sum_i (x_i^2 + y_i^2 + beta z_i^2)
    const double beta = m_parameters[1];

    double sum = 0.0;
    for (auto& p : particles) {
        const auto& r = p->getPosition();
        sum += r[0]*r[0] + r[1]*r[1] + beta*r[2]*r[2];
    }
    return -sum;
}

double CorrelatedGaussian::computeDoubleDerivative(
    std::vector<std::unique_ptr<class Particle>>& particles)
{
    const double alpha = m_parameters[0];
    const double beta  = m_parameters[1];
    const double a     = m_parameters[2];

    const unsigned int N = static_cast<unsigned int>(particles.size());
    const unsigned int D = particles[0]->getNumberOfDimensions();
    assert(D == 3);

    const double psi = evaluate(particles);
    if (psi == 0.0) {
        return 0.0;
    }

    // We use:
    // (1/Psi) sum_k nabla_k^2 Psi = sum_k [ nabla_k^2 ln Psi + |nabla_k ln Psi|^2 ]

    double totalRatio = 0.0;

    for (unsigned int k = 0; k < N; ++k) {
        const auto& rk = particles[k]->getPosition();

        // Gradient of ln(g_k)
        double gx = -2.0 * alpha * rk[0];
        double gy = -2.0 * alpha * rk[1];
        double gz = -2.0 * alpha * beta * rk[2];

        // Laplacian of ln(g_k)
        double lapLog = -2.0 * alpha * (2.0 + beta);

        // Add pair contributions from the Jastrow factor
        for (unsigned int j = 0; j < N; ++j) {
            if (j == k) continue;

            const auto& rj = particles[j]->getPosition();

            const double dx = rk[0] - rj[0];
            const double dy = rk[1] - rj[1];
            const double dz = rk[2] - rj[2];

            const double r2  = dx*dx + dy*dy + dz*dz;
            const double rij = std::sqrt(r2);

            if (rij <= a) {
                return 0.0;
            }

            // grad_k ln f(r_kj)
            const double coeff = a / (r2 * (rij - a));
            gx += coeff * dx;
            gy += coeff * dy;
            gz += coeff * dz;

            // laplacian_k ln f(r_kj)
            lapLog += - (a*a) / (r2 * (rij - a) * (rij - a));
        }

        totalRatio += lapLog + gx*gx + gy*gy + gz*gz;
    }

    return psi * totalRatio;
}