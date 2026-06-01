#pragma once
#include <memory>
#include <vector>

#include "WaveFunctions/nn_envelope.h"
#include "WaveFunctions/wavefunction.h"
#include "sampler.h"

class NNsampler : Sampler {
public:
    NNsampler(
        unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        unsigned int numberOfParameters,
        double stepLength,
        unsigned int numberOfMetropolisSteps,
        WaveFunction& wf_train
    );

    void sample(bool acceptedStep, class System* system, std::ofstream* outfile = nullptr) override;
    void computeAverages();
    void printOutputToTerminal();

    double get_K() const;
    double get_Kerr() const;
    // double getEnergy() const;
    // double getEnergyerr() const;
    double getAcceptanceRatio() const;

    // std::vector<double> get_dEdW() const;
    std::vector<double> get_dKdW() const;

    void logOutput(std::ofstream& outs);

private:
    WaveFunction& m_wf_train;

    double m_acceptanceRatio = 0;
    double m_B = 0;
    double m_B2 = 0;
    double m_cumulativeB = 0;
    double m_cumulativeB2 = 0;
    double m_K = 0;
    double m_K2 = 0;
    double m_Kerr = 0;

    const double c_eps = 1e-12; // against numerical errors

    // <O>
    std::vector<double> m_OW;
    std::vector<double> m_cumulativeOW;
    // <B O>
    std::vector<double> m_cumulativeBOW;
    std::vector<double> m_BOW;
    // <B2 O>
    std::vector<double> m_cumulativeB2OW;
    std::vector<double> m_B2OW;
};