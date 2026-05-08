#pragma once

#include <memory>
#include <vector>


class System {
public:
    System(
            std::unique_ptr<class Hamiltonian> hamiltonian,
            std::unique_ptr<class WaveFunction> waveFunction,
            std::unique_ptr<class MonteCarlo> solver,
            std::vector<std::unique_ptr<class Particle>> particles);

    unsigned int runEquilibrationSteps(
            double stepLength,
            unsigned int numberOfEquilibrationSteps);

    std::unique_ptr<class Sampler> runMetropolisSteps(
            double stepLength,
            unsigned int numberOfMetropolisSteps,
            bool storeEnergyHistory = false,
            bool storeDensityHist = false,
            bool densityOnly = false,
            unsigned int densityBins = 200,
            double densityZMax = 4.0,
            double densityRhoMax = 4.0
        );

    std::unique_ptr<class RBMSampler> runRBMMetropolisSteps(
            double stepLength,
            unsigned int numberOfMetropolisSteps,
            unsigned int numberOfHidden,
            bool storeEnergyHistory = false
    );

    std::unique_ptr<class NNsampler> runMetropolisSteps_NN(double stepParameter,
        unsigned int numberOfMetropolisSteps, WaveFunction& wf_train);

    // Helper used by Sampler
    double computeLocalEnergy();

    // Helper for printing
    const std::vector<double>& getWaveFunctionParameters();

    // Helper used to compute the VMC energy gradient
    class WaveFunction& getWaveFunction();
    std::vector<std::unique_ptr<class Particle>>& getParticles();

private:
    unsigned int m_numberOfParticles = 0;
    unsigned int m_numberOfDimensions = 0;

    std::unique_ptr<class Hamiltonian> m_hamiltonian;
    std::unique_ptr<class WaveFunction> m_waveFunction;
    std::unique_ptr<class MonteCarlo> m_solver;
    std::vector<std::unique_ptr<class Particle>> m_particles;
};

