#pragma once
#include <memory>
#include <vector>
#include <chrono>

#include "sampler.h"

/**
 * @brief Sampler dedicated to computing the energy of the quantum system.
 * * Accumulates the local energy at each Metropolis step to calculate 
 * the expectation value of the Hamiltonian, its variance, and the standard error.
 */
class EnergySampler : Sampler {
public:
    EnergySampler(
        unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        unsigned int numberOfParameters,
        double stepLength,
        unsigned int numberOfMetropolisSteps);


    void sample(bool acceptedStep, class System* system, std::ofstream* energiesOut = nullptr);
    void printOutputToTerminal(class System& system);
    void printOutputToFile(class System& system, std::ofstream& outs);
    void logOutput(const std::vector<double>& params, std::ofstream& outs);
    void logOutput(std::ofstream& outs, std::vector<double> additional_log = {});
    void computeAverages();
    double getEnergy() { return m_energy; }
    double getError() { return m_error; }
    double getAcceptanceRatio() { return (double)m_numberOfAcceptedSteps / (double)m_numberOfMetropolisSteps; }
    double getCovariance(unsigned int param_idx) { return m_covariance[param_idx]; }
    std::vector<double> get_dEdW() const;

private:
    double m_energy = 0;
    double m_energySQ = 0;
    double m_variance = 0;
    std::vector<double> m_covariance;
    std::vector<double> m_opO;
    double m_error = 0;
    double m_cumulativeEnergy = 0;
    double m_cumulativeEnergySQ = 0;
};
