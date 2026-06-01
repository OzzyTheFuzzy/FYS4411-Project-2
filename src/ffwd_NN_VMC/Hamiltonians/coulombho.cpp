#include <memory>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <limits>

#include "../common.h"
#include "coulombho.h"
#include "../particle.h"
#include "../WaveFunctions/wavefunction.h"
#include "../WaveFunctions/nn_envelope.h"
#include "../WaveFunctions/wavefunctioncache.h"

using namespace CommonUtils;

CoulombHO::CoulombHO(double omega)
    : CoulombHO(omega, omega) {}

CoulombHO::CoulombHO(double omega, double omega_z)
    : CoulombHO(omega, omega_z, 0.0043) {}

CoulombHO::CoulombHO(double omega, double omega_z, double maxStrength)
    : CoulombHO(omega, omega_z, maxStrength, 1) {}

CoulombHO::CoulombHO(double omega, double omega_z, double maxStrength, double percStrength) {
    if (omega <= 0) throw std::invalid_argument("omega needs to be a positive value");
    if (omega_z <= 0) throw std::invalid_argument("omega_z needs to be a positive value");
    if (!(0 <= percStrength && percStrength <= 1)) {
        std::cout << percStrength << std::endl;
        throw std::invalid_argument("percStrength needs to be a value within [0,1].");
    }

    m_omega = omega;
    m_omega_z = omega_z;
    m_maxStrength = maxStrength;
    m_percStrength = percStrength;
}

double CoulombHO::computeLocalEnergy(
    class WaveFunction& waveFunction,
    std::vector<std::unique_ptr<class Particle>>& particles
) {
    double kineticEnergy, potentialEnergy = 0;
    if (m_maxStrength * m_percStrength != 0) {
        double dist = 0;
        for (unsigned int i = 0; i < particles.size(); i++) {
            for (unsigned int k = 0; k < i; k++) {
                dist = 0;
                for (unsigned int j = 0; j < particles[0]->getNumberOfDimensions(); j++) {
                    dist += sq(particles[i]->getPosition()[j] - particles[k]->getPosition()[j]);
                }
                dist = sqrt(dist);
                potentialEnergy += m_maxStrength * m_percStrength / (dist + c_eps);
                // std::cout << "DEBUG: added " << m_maxStrength * m_percStrength << " to the potential" << std::endl;
            }
        }
    }

    auto* ptr = dynamic_cast<NN_envelope*>(&waveFunction);
    if (ptr == nullptr) {   // if wf is not a neural network
        kineticEnergy = -0.5 * waveFunction.computeNumericalDoubleDerivative(particles) / waveFunction.evaluate(particles);
    }
    else {  // if wf is a neural network
        kineticEnergy = -0.5 * waveFunction.computeDoubleDerivative(particles); // already normalized
    }

    double sum = 0;
    for (unsigned int i = 0; i < particles.size(); i++) {
        for (unsigned int j = 0; j < particles[0]->getNumberOfDimensions(); j++) {
            if (j == 2) {
                sum += sq(m_omega_z * particles[i]->getPosition()[j]);
            }
            else {
                sum += sq(m_omega * particles[i]->getPosition()[j]);
            }
        }
    }
    potentialEnergy += 0.5 * sum; // * m

    return kineticEnergy + potentialEnergy;
}

double CoulombHO::computeLocalEnergy(
    class WaveFunction& waveFunction,
    std::vector<std::unique_ptr<class Particle>>& particles,
    WaveFunctionCache& cache
) {
    double kineticEnergy, potentialEnergy = 0;
    if (m_maxStrength * m_percStrength != 0) {
        double dist = 0;
        for (unsigned int i = 0; i < particles.size(); i++) {
            for (unsigned int k = 0; k < i; k++) {
                dist = 0;
                for (unsigned int j = 0; j < particles[0]->getNumberOfDimensions(); j++) {
                    dist += sq(particles[i]->getPosition()[j] - particles[k]->getPosition()[j]);
                }
                dist = sqrt(dist);
                potentialEnergy += m_maxStrength * m_percStrength / (dist + c_eps);
                // std::cout << "DEBUG: added " << m_maxStrength * m_percStrength << " to the potential" << std::endl;
            }
        }
    }

    if (waveFunction.hasAnalyticalDerivative() && m_analytic_ifAvailable) {
        kineticEnergy = -0.5 * waveFunction.computeDoubleDerivative(particles) / waveFunction.evaluate(particles);
    }
    else {
        // the following commented line is deprecated
        // kineticEnergy = -0.5 * waveFunction.computeNumericalDoubleDerivative(particles) / waveFunction.evaluate(particles);
        kineticEnergy = -0.5 * cache.computeNumericalLaplacian(particles, waveFunction);
    }

    double sum = 0;
    for (unsigned int i = 0; i < particles.size(); i++) {
        for (unsigned int j = 0; j < particles[0]->getNumberOfDimensions(); j++) {
            if (j == 2) {
                sum += sq(m_omega_z * particles[i]->getPosition()[j]);
            }
            else {
                sum += sq(m_omega * particles[i]->getPosition()[j]);
            }
        }
    }
    potentialEnergy += 0.5 * sum; // * m

    return kineticEnergy + potentialEnergy;
}

void CoulombHO::set_percStrength(double percStrength) {
    if (!(0 <= percStrength && percStrength <= 1))
        throw std::invalid_argument("percStrength needs to be a value within [0,1]");
    m_percStrength = percStrength;
}

double CoulombHO::get_interaction_strength() {
    return m_percStrength * m_maxStrength;
}