#pragma once

#include <memory>

#include "wavefunction.h"

class EllipticGaussian : public WaveFunction {
    /* assumes 3D case */
public:
    EllipticGaussian(double alpha, double beta);
    double evaluate(std::vector<std::unique_ptr<class Particle>>& particles);
    double evaluateLn(std::vector<std::unique_ptr<class Particle>>& particles);
    double computeParticleLn(std::vector<std::unique_ptr<Particle>>& particles,
        unsigned int particle_idx);
    double computeParamDerivativeLn(std::vector<std::unique_ptr<class Particle>>& particles,
        unsigned int param_idx);
    bool hasAnalyticalDerivative() override { return true; }
    std::vector<double> lowerBounds() const override { return { 1e-3, 0.1 }; }
    std::vector<double> upperBounds() const override { return { 1.0, 5.0 }; }
    double computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles);
    std::vector<double> computeQuantumForce(
        std::vector<std::unique_ptr<class Particle>>& particles, unsigned int particle_idx);
private:
    const unsigned int m_NDIM = 3;
};
