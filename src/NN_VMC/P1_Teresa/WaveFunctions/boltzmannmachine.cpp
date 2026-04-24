#include <memory>
#include <cmath>
#include <cassert>
#include <vector>

#include "boltzmannmachine.h"
#include "../particle.h"

BoltzmannMachine::BoltzmannMachine(
    std::vector<std::vector<double>> a,
    std::vector<double> b,
    std::vector<std::vector<std::vector<double>>> w
) : m_a(a), m_b(b), m_w(w), m_nParticles(a.size()),
m_nDim(a[0].size()), m_nHidden(b.size()) {}

std::vector<double> BoltzmannMachine::QFac(std::vector<std::unique_ptr<class Particle>>& particles) {
    std::vector<double> Q(m_nHidden);

    for (int i = 0; i < m_nHidden; i++) {
        for (int j = 0; j < m_nParticles; j++) {
            for (int k = 0; k < m_nDim; k++) {
                Q[i] = matr_mult(particles.getPosition(), m_w[j]);
            }
        }
    }
}

double BoltzmannMachine::evaluate(std::vector<std::unique_ptr<class Particle>>& particles) {
    double sigma = 1;
    double sig2 = sigma * sigma;
    double Psi1 = 0;
    double Psi2 = 1;

    double sum = 0.0;
    for (auto& p : particles) {
        const auto& r = p->getPosition();
        sum += r[0] * r[0] + r[1] * r[1] + beta * r[2] * r[2];
    }

    return std::exp(-alpha * sum);
}

double BoltzmannMachine::computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles) {
    const double alpha = m_parameters[0];
    const double beta = m_parameters[1];

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

        sum += lapLog + gx * gx + gy * gy + gz * gz;
    }

    return psi * sum;
}

double BoltzmannMachine::computeLogDerivativeAlpha(std::vector<std::unique_ptr<class Particle>>& particles) {
    const double beta = m_parameters[1];

    double sum = 0.0;
    for (auto& p : particles) {
        const auto& r = p->getPosition();
        sum += r[0] * r[0] + r[1] * r[1] + beta * r[2] * r[2];
    }

    return -sum;
}