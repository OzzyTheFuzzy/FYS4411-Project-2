#include <memory>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cassert>

#include "../common.h"
#include "ellipticgaussian.h"
#include "wavefunction.h"
#include "../system.h"
#include "../particle.h"

using namespace CommonUtils;

EllipticGaussian::EllipticGaussian(double alpha, double beta)
    : WaveFunction(2, { alpha, beta }) {
    if (alpha < 0 || beta < 0) throw std::invalid_argument("alpha and beta must be non-negative");
}

double EllipticGaussian::evaluate(std::vector<std::unique_ptr<class Particle>>& particles) {
    return exp(evaluateLn(particles));
}

double EllipticGaussian::evaluateLn(std::vector<std::unique_ptr<class Particle>>& particles) {
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

double EllipticGaussian::computeParticleLn(
    std::vector<std::unique_ptr<Particle>>& particles,
    unsigned int particle_idx) {
    double sum = 0;
    for (unsigned int j = 0; j < m_NDIM; j++) {
        if (j == 2) {
            sum += m_parameters[1] * sq(particles[particle_idx]->getPosition()[j]);
        }
        else {
            sum += sq(particles[particle_idx]->getPosition()[j]);
        }
    }
    return -m_parameters[0] * sum;
}

double EllipticGaussian::computeParamDerivativeLn(
    std::vector<std::unique_ptr<class Particle>>& particles, unsigned int param_idx) {
    long double sum = 0;

    if (param_idx == 0) {       // derivative wrt alpha
        // sum all coordinates squared:  Σ_i [x_i² + y_i² + βz_i²]
        for (unsigned int i = 0; i < particles.size(); i++) {
            for (unsigned int j = 0; j < m_NDIM; j++) {
                if (j == 2)
                    sum += m_parameters[1] * sq(particles[i]->getPosition()[j]);
                else
                    sum += sq(particles[i]->getPosition()[j]);
            }
        }

        return -sum;
    }
    else if (param_idx == 1) {  // derivative wrt beta
        // sum Σ_i z_i²
        for (unsigned int i = 0; i < particles.size(); i++) {
            sum += sq(particles[i]->getPosition()[2]);
        }
        return -m_parameters[0] * sum;  // -α * sum
    }

    throw std::invalid_argument("Invalid param_idx.");
}

double EllipticGaussian::computeDoubleDerivative(
    std::vector<std::unique_ptr<class Particle>>& particles) {
    /* All wave functions need to implement this function, so you need to
     * find the double derivative analytically. Note that by double derivative,
     * we actually mean the sum of the Laplacians with respect to the
     * coordinates of each particle.
     *
     * This quantity is needed to compute the (local) energy (consider the
     * Schrödinger equation to see how the two are related).
     */

    double alpha = m_parameters[0];
    double beta = m_parameters[1];
    double tot_eval = evaluate(particles);

    double sum_over_particles = 0;
    for (unsigned int i = 0; i < particles.size(); i++) {
        double rad_sq = 0;       // x² + y² + βz²   (for the linear alpha term)
        double rad_sq2 = 0;      // x² + y² + β²z²  (for the quadratic alpha term)

        for (unsigned int j = 0; j < m_NDIM; j++) {
            if (j == 2) {
                rad_sq += beta * sq(particles[i]->getPosition()[j]);
                rad_sq2 += sq(beta * particles[i]->getPosition()[j]);
            }
            else {
                rad_sq += sq(particles[i]->getPosition()[j]);
                rad_sq2 += sq(particles[i]->getPosition()[j]);
            }
        }

        double phi_i = exp(-alpha * rad_sq);
        double lapl_term = (-2 * alpha * (2 + beta) + 4 * sq(alpha) * rad_sq2) * phi_i;

        double prod_term = tot_eval / phi_i;

        sum_over_particles += lapl_term * prod_term;
    }
    return sum_over_particles;
}

std::vector<double> EllipticGaussian::computeQuantumForce(
    std::vector<std::unique_ptr<class Particle>>& particles,
    unsigned int particle_idx) {
    double alpha = m_parameters[0];
    double beta = m_parameters[1];
    std::vector<double> qForce = std::vector<double>(m_NDIM);

    for (unsigned int i = 0; i < m_NDIM; i++) {
        qForce[i] = -4 * alpha * particles[particle_idx]->getPosition()[i];
        if (i == 2)
            qForce[i] *= beta;
    }

    return qForce;
}