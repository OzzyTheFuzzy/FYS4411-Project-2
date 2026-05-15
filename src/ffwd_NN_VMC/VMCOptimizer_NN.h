#pragma once
#include <vector>
#include <fstream>
#include <string>

#include "mcengine.h"
#include "Samplers/energysampler.h"
#include "Samplers/NNsampler.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/ellipticgaussian.h"
#include "WaveFunctions/repulsiveellipticgaussian.h"
#include "WaveFunctions/nn_envelope.h"

class VMCOptimizer_NN {
public:
    /// @brief Type for the Hamiltonian factory.
    using HamiltonianFactory = std::function<std::unique_ptr<class Hamiltonian>()>;
    /// @brief Type for the Solver factory (accepts a random generator).
    using SolverFactory = std::function<std::unique_ptr<class MonteCarlo>(std::unique_ptr<class Random>)>;
    
    VMCOptimizer_NN(
        unsigned int numberOfDimensions,
        unsigned int numberOfParticles,
        unsigned int numberOfEquilibrationSteps,
        double timeStep,
        std::unique_ptr<class Hamiltonian> hamiltonian,
        SolverFactory solverFactory,
        int seed,
        int Nhid,
        double helpDecay,
        int optEquil,
        int nSamples,
        int nPretrainSteps,
        int nEnergySteps,
        int nAdiabSteps,
        double maxStrength,
        double lr,
        double Adam_tol,
        std::ofstream* logfile = nullptr,
        std::ofstream* outfile = nullptr,
        std::ofstream* paramsfile = nullptr
    );

    std::vector<double> optimize(std::unique_ptr<WaveFunction> wf_train);

    // deprecated
    void resetPos(std::vector<std::unique_ptr<class Particle>>& particles, std::vector<std::vector<double>>& pos);
private:
    double computeMC(const std::vector<double>& params, std::vector<double>& grad);
    unsigned int m_numberOfDimensions;
    unsigned int m_numberOfParticles;
    unsigned int m_numberOfEquilibrationSteps;
    double m_timeStep;
    std::unique_ptr<class Hamiltonian> m_hamiltonian;
    SolverFactory m_solverFactory;
    int m_seed;
    int m_Nhid;
    double m_helpDecay;
    int m_optEquil;
    int m_nSamples;
    int m_nPretrainSteps;
    int m_nEnergySteps;
    int m_nAdiabSteps;
    double m_maxStrength;
    double m_lr;
    double m_Adam_ktol;
    std::ofstream* m_logfile;
    std::ofstream* m_outfile;
    std::ofstream* m_paramsfile;
    unsigned int m_mcCount = 0;
};