#pragma once
#include <memory>
#include <vector>

#include "wavefunction.h"

class CorrelatedGaussian : public WaveFunction {
public:
    CorrelatedGaussian(double alpha, double beta, double a);

    double evaluate(std::vector<std::unique_ptr<class Particle>>& particles) override;
    double computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles) override;
    double computeLogDerivativeAlpha(std::vector<std::unique_ptr<class Particle>>& particles) override;

private:
    double pairDistance(
        const std::vector<double>& ri,
        const std::vector<double>& rj
    ) const;
};