// -----------------------------------------------------------------------------
// BoltzmannMachine wave function
//
// Visible variables:
//   r[p][d]   = coordinate d of particle p
//   a[p][d]   = visible bias for coordinate d of particle p
//
// Hidden variables:
//   b[h]      = hidden bias for hidden node h
//   w[p][d][h]= coupling between visible coordinate (p,d) and hidden node h
//
// Dimensions:
//   p = 0,...,N_particles-1
//   d = 0,...,N_dim-1
//   h = 0,...,N_hidden-1
//
// RBM ansatz (Gaussian-binary, with sigma = 1 here):
//
//   Q_h = b_h + sum_{p,d} r_{p,d} w_{p,d,h}
//
//   Psi(r) = exp[ -1/2 sum_{p,d} (r_{p,d} - a_{p,d})^2 ]
//            * prod_h ( 1 + exp(Q_h) )
//
// This is the trial wave function used inside the VMC framework.
// -----------------------------------------------------------------------------

#include <memory>
#include <cmath>
#include <cassert>
#include <vector>
#include <stdexcept>

#include "boltzmannmachine.h"
#include "../particle.h"

BoltzmannMachine::BoltzmannMachine(
    std::vector<std::vector<double>> a,
    std::vector<double> b,
    std::vector<std::vector<std::vector<double>>> w
) : m_a(a), m_b(b), m_w(w), m_nParticles(a.size()),
m_nDim(a[0].size()), m_nHidden(b.size()) {
    // Number of trainable RBM parameters:
    //
    //   a[p][d]      -> N_particles * D
    //   b[h]         -> N_hidden
    //   w[p][d][h]   -> N_particles * D * N_hidden
    m_numberOfParameters =
        m_nParticles * m_nDim
        + m_nHidden
        + m_nParticles * m_nDim * m_nHidden;
}

// Compute the hidden-layer activation
//
//   Q_h = b_h + sum_{p,d} r_{p,d} w_{p,d,h}
//
// Returns a vector Q of size m_nHidden.
// Each Q[h] corresponds to one hidden node.
std::vector<double> BoltzmannMachine::QFac(std::vector<std::unique_ptr<class Particle>>& particles) {
    std::vector<double> Q(m_nHidden, 0.0);

    for (int h =0; h < m_nHidden; h++) {
        Q[h] = m_b[h];
        for (int p = 0; p < m_nParticles; p++) {
            const auto& r = particles[p]->getPosition();
            for (int d = 0; d < m_nDim; d++) {
                Q[h] += r[d] * m_w[p][d][h];
            }
        }
    }
    return Q;
}

// Evaluate the RBM wave function
//
//   Psi(r) = Psi1 * Psi2
//
// with
//   Psi1 = exp[ -1/(2 sigma^2) sum_{p,d} (r_{p,d} - a_{p,d})^2 ]
//   Psi2 = prod_h ( 1 + exp(Q_h) )
//
// Here sigma = 1.0, so sigma^2 = 1.
double BoltzmannMachine::evaluate(std::vector<std::unique_ptr<class Particle>>& particles) {
    double sigma = 1.0;
    double sig2 = sigma * sigma;

    double Psi1 = 0.0;
    double Psi2 = 1.0;

    std::vector<double> Q = QFac(particles);

    for (int p = 0; p < m_nParticles; p++) {
        const auto& r = particles[p]->getPosition();
        for (int d = 0; d < m_nDim; d++) {
            Psi1 += (r[d] - m_a[p][d]) * (r[d] -m_a[p][d]);
        }
    }

    for (int h = 0; h < m_nHidden; h++) {
        Psi2 *= (1.0 + std::exp(Q[h]));
    }

    Psi1 = std::exp(-Psi1 / (2.0 * sig2));

    return Psi1 * Psi2;
}

// Compute the Laplacian of the RBM wave function:
//
//   nabla^2 Psi = Psi * sum_i [ d^2(ln Psi)/dX_i^2 + (d(ln Psi)/dX_i)^2 ]
//
// This is needed for the kinetic term in the local energy:
//
//   E_L = -1/2 * (nabla^2 Psi / Psi) + V(R)
double BoltzmannMachine::computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles){
    const double sigma = 1.0;
    const double sig2 = sigma * sigma;
    const double sig4 = sig2 * sig2;

    const double psi = evaluate(particles);
    const std::vector<double> Q = QFac(particles);

    double sum = 0.0;

    for (int p = 0; p < m_nParticles; p++) {
        const auto& r = particles[p]->getPosition();

        for (int d = 0; d < m_nDim; d++) {
            const double Xi = r[d];

            double first = -(Xi - m_a[p][d]) / sig2;
            double second = -1.0 / sig2;

            for (int h = 0; h < m_nHidden; h++) {
                const double expMinusQ = std::exp(-Q[h]);
                const double denom = 1.0 + expMinusQ;
                const double logistic = 1.0 / denom;

                first += (m_w[p][d][h] / sig2) * logistic;
                second += (m_w[p][d][h] * m_w[p][d][h] / sig4)
                          * expMinusQ / (denom * denom);
            }

            sum += second + first * first;
        }
    }

    return psi * sum;
}

// RBM quantum force:
//   F_{p,d} = 2 * d ln Psi / d X_{p,d}
// with
//   d ln Psi / d X_{p,d}
//      = -(X_{p,d} - a_{p,d}) / sigma^2
//        + sum_h w_{p,d,h} sigmoid(Q_h) / sigma^2
// where
//   sigmoid(Q_h) = 1 / (1 + exp(-Q_h)).
std::vector<double> BoltzmannMachine::computeQuantumForce(std::vector<std::unique_ptr<class Particle>>& particles, unsigned int particleIndex) {
    const double sigma = 1.0;
    const double sig2 = sigma * sigma;

    const std::vector<double> Q = QFac(particles);
    const std::vector<double>& r = particles[particleIndex]->getPosition();

    std::vector<double> force(m_nDim, 0.0);

    for (int d = 0; d < m_nDim; d++) {
        const double Xpd = r[d];
        double dlnpsi = -(Xpd - m_a[particleIndex][d]) / sig2;

        for (int h = 0; h < m_nHidden; h++) {
            const double sigmoid = 1.0 / (1.0 + std::exp(-Q[h]));
            dlnpsi += (m_w[particleIndex][d][h] / sig2) * sigmoid;
        }

        force[d] = 2.0 * dlnpsi;
    }

    return force;
}

// Compute logarithmic derivatives of the RBM wave function
// with respect to all variational parameters.
//
//   (1/Psi) dPsi/da[p][d]   = (r[p][d] - a[p][d]) / sigma^2
//   (1/Psi) dPsi/db[h]      = 1 / (1 + exp(-Q[h]))
//   (1/Psi) dPsi/dw[p][d][h]= r[p][d] / sigma^2 * 1/(1 + exp(-Q[h]))
//
// These derivatives are used to build the VMC energy gradient.
RBMParameterDerivatives BoltzmannMachine::computeParameterDerivatives(
    std::vector<std::unique_ptr<class Particle>>& particles)
{
    const double sigma = 1.0;
    const double sig2 = sigma * sigma;

    const std::vector<double> Q = QFac(particles);

    RBMParameterDerivatives der;
    der.dA.assign(m_nParticles, std::vector<double>(m_nDim, 0.0));
    der.dB.assign(m_nHidden, 0.0);
    der.dW.assign(
        m_nParticles,
        std::vector<std::vector<double>>(m_nDim, std::vector<double>(m_nHidden, 0.0))
    );

    std::vector<double> logistic(m_nHidden, 0.0);
    for (int h = 0; h < m_nHidden; h++) {
        logistic[h] = 1.0 / (1.0 + std::exp(-Q[h]));
        der.dB[h] = logistic[h];
    }

    for (int p = 0; p < m_nParticles; p++) {
        const auto& r = particles[p]->getPosition();

        for (int d = 0; d < m_nDim; d++) {
            der.dA[p][d] = (r[d] - m_a[p][d]) / sig2;

            for (int h = 0; h < m_nHidden; h++) {
                der.dW[p][d][h] = (r[d] / sig2) * logistic[h];
            }
        }
    }

    return der;

}
