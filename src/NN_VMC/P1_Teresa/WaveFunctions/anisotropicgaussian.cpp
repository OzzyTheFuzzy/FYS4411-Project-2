#include <memory>
#include <cmath>
#include <cassert>
#include <vector>

#include "anisotropicgaussian.h"
#include "../particle.h"

AnisotropicGaussian::AnisotropicGaussian(double alpha, double beta)
{
    assert(alpha >= 0.0);
    assert(beta > 0.0);

    m_numberOfParameters = 2;
    m_parameters = {alpha, beta};
}

double AnisotropicGaussian::evaluate(std::vector<std::unique_ptr<class Particle>>& particles)
{
    const double alpha = m_parameters[0];
    const double beta  = m_parameters[1];

    double sum = 0.0;
    for (auto& p : particles) {
        const auto& r = p->getPosition();
        sum += r[0]*r[0] + r[1]*r[1] + beta*r[2]*r[2];
    }

    return std::exp(-alpha * sum);
}

double AnisotropicGaussian::computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles)
{
    const double alpha = m_parameters[0];
    const double beta  = m_parameters[1];

    const double psi = evaluate(particles);

    double sum = 0.0;
    for (auto& p : particles) {
        const auto& r = p->getPosition();

        // grad ln Psi = (-2ax, -2ay, -2ab z)
        const double gx = -2.0 * alpha * r[0];
        const double gy = -2.0 * alpha * r[1];
        const double gz = -2.0 * alpha * beta * r[2];

        // lap ln Psi = -2a(2+beta)
        const double lapLog = -2.0 * alpha * (2.0 + beta);

        sum += lapLog + gx*gx + gy*gy + gz*gz;
    }

    return psi * sum;
}

double AnisotropicGaussian::computeLogDerivativeAlpha(std::vector<std::unique_ptr<class Particle>>& particles)
{
    const double beta = m_parameters[1];

    double sum = 0.0;
    for (auto& p : particles) {
        const auto& r = p->getPosition();
        sum += r[0]*r[0] + r[1]*r[1] + beta*r[2]*r[2];
    }

    return -sum;
}