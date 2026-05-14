#include <memory>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <limits>

#include "../common.h"
#include "boltzmannmachine.h"
#include "wavefunction.h"
#include "../system.h"
#include "../particle.h"

using namespace CommonUtils;

BoltzmannMachine::BoltzmannMachine(double alpha, double beta, double rep_a)
    : WaveFunction(2, { alpha, beta }), m_rep_a(rep_a) {
    if (alpha < 0 || beta < 0 || rep_a < 0) throw std::invalid_argument("alpha, beta and rep_a must be non-negative");
}

double BoltzmannMachine::evaluate(std::vector<std::unique_ptr<class Particle>>& particles) {
    double prod = 1;
    double dist = 0;
    for (unsigned int i = 0; i < particles.size(); i++) {
        for (unsigned int k = 0; k < i; k++) {
            dist = 0;
            for (unsigned int j = 0; j < particles[0]->getNumberOfDimensions(); j++) {
                dist += sq(particles[i]->getPosition()[j] - particles[k]->getPosition()[j]);
            }
            dist = sqrt(dist);
            if (dist <= m_rep_a)
                return 0;
            prod *= (1 - m_rep_a / dist);
        }
    }

    return exp(evaluateLn_noInteraction(particles)) * prod;
    // or, with less accurate results:
    // return exp(evaluateLn(particles));
}

double BoltzmannMachine::evaluateLn_noInteraction(std::vector<std::unique_ptr<class Particle>>& particles) {
    long double sum = 0;

    // sum all coordinates squared:  Σ_i [x_i² + y_i² + βz_i²]
    for (unsigned int i = 0; i < particles.size(); i++) {
        for (unsigned int j = 0; j < m_NDIM; j++) {
            if (j == 2)
                sum += m_parameters[1] * sq(particles[i]->getPosition()[j]);
            else
                sum += sq(particles[i]->getPosition()[j]);
        }
    }

    return -m_parameters[0] * sum;  // -α * sum
}

double BoltzmannMachine::evaluateLn(std::vector<std::unique_ptr<class Particle>>& particles) {
    long double intersum = 0; // sum of ln f
    double dist = 0;
    for (unsigned int i = 0; i < particles.size(); i++) {
        for (unsigned int k = 0; k < i; k++) {
            dist = 0;
            for (unsigned int j = 0; j < particles[0]->getNumberOfDimensions(); j++) {
                dist += sq(particles[i]->getPosition()[j] - particles[k]->getPosition()[j]);
            }
            dist = sqrt(dist);
            if (dist <= m_rep_a)
                return -std::numeric_limits<double>::infinity();
            intersum += log(1 - m_rep_a / dist);
        }
    }

    return evaluateLn_noInteraction(particles) + intersum;  // -α * sum + sum of ln f
}

double BoltzmannMachine::computeParticleLn(
    std::vector<std::unique_ptr<Particle>>& particles,
    unsigned int particle_idx) {
    // single-particle term
    double sum = 0;
    for (unsigned int j = 0; j < m_NDIM; j++) {
        if (j == 2) {
            sum += m_parameters[1] * sq(particles[particle_idx]->getPosition()[j]);
        }
        else {
            sum += sq(particles[particle_idx]->getPosition()[j]);
        }
    }
    double single_part = -m_parameters[0] * sum;

    // Jastrow terms for all pairs involving particle_idx
    double jastrow_part = 0;
    for (unsigned int k = 0; k < particles.size(); k++) {
        if (k == particle_idx) continue;
        double dist = 0.0;
        for (unsigned int j = 0; j < m_NDIM; j++) {
            dist += sq(particles[particle_idx]->getPosition()[j]
                - particles[k]->getPosition()[j]);
        }
        dist = sqrt(dist);
        if (dist <= m_rep_a) return -std::numeric_limits<double>::infinity();
        jastrow_part += log(1.0 - m_rep_a / dist);
    }
    return single_part + jastrow_part;
}