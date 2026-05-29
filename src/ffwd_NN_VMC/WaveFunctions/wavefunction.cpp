#include <memory>
#include <cmath>
#include <cassert>
#include <iostream>

#include "../common.h"
#include "wavefunction.h"
#include "../system.h"
#include "../particle.h"

using namespace CommonUtils;

double WaveFunction::computeNumericalDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles) {
    double sum = 0.0;
    double wfCurrent = evaluate(particles);
    unsigned int numberOfDimensions = particles[0]->getNumberOfDimensions();

    for (auto& particle : particles) {

        for (unsigned int d = 0; d < numberOfDimensions; d++) {
            double h = 1e-4 * std::max(1.0, std::abs(particle->getPosition()[d]));
            double h2 = h * h;

            particle->adjustPosition(h, d);
            double wfPlus = evaluate(particles);

            particle->adjustPosition(-2.0 * h, d);
            double wfMinus = evaluate(particles);

            particle->adjustPosition(h, d);

            sum += (wfPlus - 2.0 * wfCurrent + wfMinus) / h2;
        }
    }

    return sum;
}

double WaveFunction::computeNumericalParamDerivativeLn(std::vector<std::unique_ptr<class Particle>>& particles, unsigned int param_idx) {
    double h = 1e-4 * std::max(1.0, std::abs(m_parameters[param_idx]));

    m_parameters[param_idx] += h;
    double lnPlus = evaluateLn(particles);

    m_parameters[param_idx] -= 2.0 * h;
    double lnMinus = evaluateLn(particles);

    m_parameters[param_idx] += h;  // restore

    return (lnPlus - lnMinus) / (2.0 * h);
}

std::vector<double> WaveFunction::computeNumericalQuantumForce(std::vector<std::unique_ptr<class Particle>>& particles, unsigned int particle_idx) {
    unsigned int numberOfDimensions = particles[particle_idx]->getNumberOfDimensions();
    std::vector<double> qForce(numberOfDimensions);

    for (unsigned int d = 0; d < numberOfDimensions; d++) {
        double h = 1e-4 * std::max(1.0, std::abs(particles[particle_idx]->getPosition()[d]));

        particles[particle_idx]->adjustPosition(h, d);
        double lnPlus = evaluateLn(particles);

        particles[particle_idx]->adjustPosition(-2.0 * h, d);
        double lnMinus = evaluateLn(particles);

        particles[particle_idx]->adjustPosition(h, d);  // restore

        qForce[d] = 2.0 * (lnPlus - lnMinus) / (2.0 * h);
    }

    return qForce;
}

std::vector<double> WaveFunction::computeLogParDer_vect(std::vector<std::unique_ptr<class Particle>>& particles) {
    std::vector<double> v(m_numberOfParameters);
    for (int i = 0; i < m_numberOfParameters; i++) {
        v[i] = computeParamDerivativeLn(particles, i);
    }
    return v;
}