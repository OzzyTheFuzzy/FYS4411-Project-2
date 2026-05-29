#pragma once
#include <vector>
#include <fstream>
#include <string>

#include "mcengine.h"
#include "config.h"
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
        const runConfig& cfg,
        std::unique_ptr<class Hamiltonian> hamiltonian,
        SolverFactory solverFactory,
        ActivationFunc actFun,
        std::ofstream* logfile = nullptr,
        std::ofstream* outfile = nullptr,
        std::ofstream* paramsfile = nullptr
    );

    std::vector<double> optimize(std::unique_ptr<WaveFunction> wf_train);

    // deprecated
    void resetPos(std::vector<std::unique_ptr<class Particle>>& particles, std::vector<std::vector<double>>& pos);
private:
    // double computeMC(const std::vector<double>& params, std::vector<double>& grad);
    void set_lr(torch::optim::Adam& optimizer, double new_lr);
    bool checkPlateau(double current_K, double& best_K, int& patience_counter, double min_improvement, bool increase);

    runConfig m_cfg;

    std::unique_ptr<class Hamiltonian> m_hamiltonian;
    SolverFactory m_solverFactory;
    ActivationFunc m_actFun;

    std::ofstream* m_logfile;
    std::ofstream* m_outfile;
    std::ofstream* m_paramsfile;
};