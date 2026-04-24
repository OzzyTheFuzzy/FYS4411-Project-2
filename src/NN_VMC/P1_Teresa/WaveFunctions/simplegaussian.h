#pragma once

#include <memory>

#include "wavefunction.h"

class SimpleGaussian : public WaveFunction {
public:
    SimpleGaussian(double alpha);

    // Trial wave function value
    double evaluate(std::vector<std::unique_ptr<class Particle>>& particles) override;

    // Analytic Laplacian of the Gaussian wave function
    double computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles) override;
    
    // Logarithmic derivative with respect to alpha:
    // For Psi = exp(-alpha * sum r_i^2), this is just -sum r_i^2
    double computeLogDerivativeAlpha (
        std::vector<std::unique_ptr<class Particle>>& particles) override;
    
    const std::vector<double>& getParameters() const { return m_parameters; }
};
