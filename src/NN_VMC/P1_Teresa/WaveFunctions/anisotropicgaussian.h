#pragma once
#include <memory>
#include <vector>

#include "wavefunction.h"

class AnisotropicGaussian : public WaveFunction {
public:
    AnisotropicGaussian(double alpha, double beta);

    double evaluate(std::vector<std::unique_ptr<class Particle>>& particles) override;
    double computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles) override;
    double computeLogDerivativeAlpha(std::vector<std::unique_ptr<class Particle>>& particles) override;
};