#pragma once
#include <memory>
#include <vector>

#include "wavefunction.h"

// A parameter-derivative container
struct RBMParameterDerivatives {
    std::vector<std::vector<double>> dA;
    std::vector<double> dB;
    std::vector<std::vector<std::vector<double>>> dW;
};

class BoltzmannMachine : public WaveFunction {
public:
    BoltzmannMachine(
        std::vector<std::vector<double>> a,
        std::vector<double> b,
        std::vector<std::vector<std::vector<double>>> w);

    double evaluate(std::vector<std::unique_ptr<class Particle>>& particles) override;
    std::vector<double> QFac(std::vector<std::unique_ptr<class Particle>>& particles);
    double computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles) override;
    
    // New RBM-specific parameter derivatives
    RBMParameterDerivatives computeParameterDerivatives(
        std::vector<std::unique_ptr<class Particle>>& particles);

private:
    std::vector<std::vector<double>> m_a;
    std::vector<double> m_b;
    std::vector<std::vector<std::vector<double>>> m_w;
    int m_nParticles;
    int m_nDim;
    int m_nHidden;
};