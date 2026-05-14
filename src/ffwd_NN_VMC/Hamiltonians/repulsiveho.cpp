#include <memory>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <limits>

#include "../common.h"
#include "repulsiveho.h"
#include "../particle.h"
#include "../WaveFunctions/wavefunction.h"
#include "../WaveFunctions/nn_envelope.h"
#include "../WaveFunctions/wavefunctioncache.h"

using namespace CommonUtils;

RepulsiveHO::RepulsiveHO(double omega)
    : RepulsiveHO(omega, omega) {}

RepulsiveHO::RepulsiveHO(double omega, double omega_z)
    : RepulsiveHO(omega, omega_z, 0.0043) {}

RepulsiveHO::RepulsiveHO(double omega, double omega_z, double repulsive_a_factor) {
    if (omega <= 0) throw std::invalid_argument("omega needs to be a positive value");
    if (omega_z <= 0) throw std::invalid_argument("omega_z needs to be a positive value");
    if (repulsive_a_factor < 0) throw std::invalid_argument("repulsive_a_factor needs to be a non-negative value");

    m_omega = omega;
    m_omega_z = omega_z;
    m_rep_a = repulsive_a_factor / sqrt(omega);
}

RepulsiveHO::RepulsiveHO(double omega, double omega_z, double repulsive_a_factor, double hardcore_strength) {
    if (omega <= 0) throw std::invalid_argument("omega needs to be a positive value");
    if (omega_z <= 0) throw std::invalid_argument("omega_z needs to be a positive value");
    if (repulsive_a_factor < 0) throw std::invalid_argument("repulsive_a_factor needs to be a non-negative value");

    m_omega = omega;
    m_omega_z = omega_z;
    m_rep_a = repulsive_a_factor / sqrt(omega);
    m_strength = hardcore_strength;
}

double RepulsiveHO::computeLocalEnergy(
    class WaveFunction& waveFunction,
    std::vector<std::unique_ptr<class Particle>>& particles
) {
    double kineticEnergy, potentialEnergy = 0;
    if (m_rep_a != 0 && m_strength != 0) {
        // if a pair of particles is found at relative distance shorter than m_rep_a, then return +∞
        double dist = 0;
        for (unsigned int i = 0; i < particles.size(); i++) {
            for (unsigned int k = 0; k < i; k++) {
                dist = 0;
                for (unsigned int j = 0; j < particles[0]->getNumberOfDimensions(); j++) {
                    dist += sq(particles[i]->getPosition()[j] - particles[k]->getPosition()[j]);
                }
                dist = sqrt(dist);
                if (dist <= m_rep_a) {
                    if (m_strength == std::numeric_limits<double>::infinity()) {
                        return std::numeric_limits<double>::infinity();
                    }
                    else {
                        potentialEnergy += m_strength;
                    }
                }
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

double RepulsiveHO::computeLocalEnergy(
    class WaveFunction& waveFunction,
    std::vector<std::unique_ptr<class Particle>>& particles,
    WaveFunctionCache& cache
) {
    double potentialEnergy = 0;
    if (m_rep_a != 0 && m_strength != 0) {
        // if a pair of particles is found at relative distance shorter than m_rep_a, then return +∞
        double dist = 0;
        for (unsigned int i = 0; i < particles.size(); i++) {
            for (unsigned int k = 0; k < i; k++) {
                dist = 0;
                for (unsigned int j = 0; j < particles[0]->getNumberOfDimensions(); j++) {
                    dist += sq(particles[i]->getPosition()[j] - particles[k]->getPosition()[j]);
                }
                dist = sqrt(dist);
                if (dist <= m_rep_a) {
                    if (m_strength == std::numeric_limits<double>::infinity()) {
                        return std::numeric_limits<double>::infinity();
                    }
                    else {
                        potentialEnergy += m_strength;
                    }
                }
            }
        }
    }

    double kineticEnergy;
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