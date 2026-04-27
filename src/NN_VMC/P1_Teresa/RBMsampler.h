#pragma once
#include <memory>
#include <vector>

#include "WaveFunctions/boltzmannmachine.h"

class RBMSampler {
public:
    RBMSampler(
        unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        unsigned int numberOfHidden,
        double stepLength,
        unsigned int numberOfMetropolisSteps,
        bool storeEnergyHistory = false
    );

    void sample(bool acceptedStep, class System* system);
    void computeAverages();
    void printOutputToTerminal();

    double getEnergy() const;
    double getAcceptanceRatio() const;

    const std::vector<std::vector<double>>& getGradientA() const;
    const std::vector<double>& getGradientB() const;
    const std::vector<std::vector<std::vector<double>>>& getGradientW() const;

    const std::vector<double>& getEnergyHistory() const;

private:
    unsigned int m_stepNumber = 0;
    unsigned int m_numberOfMetropolisSteps = 0;
    unsigned int m_numberOfParticles = 0;
    unsigned int m_numberOfDimensions = 0;
    unsigned int m_numberOfHidden = 0;
    unsigned int m_numberOfAcceptedSteps = 0;

    double m_energy = 0.0;
    double m_cumulativeEnergy = 0.0;
    double m_stepLength = 0.0;

    bool m_storeEnergyHistory = false;
    std::vector<double> m_energyHistory;

    // <O>
    std::vector<std::vector<double>> m_cumulativeDA;
    std::vector<double> m_cumulativeDB;
    std::vector<std::vector<std::vector<double>>> m_cumulativeDW;

    // <E O>
    std::vector<std::vector<double>> m_cumulativeEDA;
    std::vector<double> m_cumulativeEDB;
    std::vector<std::vector<std::vector<double>>> m_cumulativeEDW;

    // Final gradients
    std::vector<std::vector<double>> m_gradientA;
    std::vector<double> m_gradientB;
    std::vector<std::vector<std::vector<double>>> m_gradientW;
};