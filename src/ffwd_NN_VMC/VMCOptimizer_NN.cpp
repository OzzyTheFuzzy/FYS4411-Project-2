#include <iostream>
#include <iomanip>
#include <nlopt.hpp>

#include "VMCOptimizer_NN.h"
#include "system.h"
#include "particle.h"
#include "Samplers/energysampler.h"
#include "Samplers/NNsampler.h"
#include "Hamiltonians/repulsiveho.h"
#include "WaveFunctions/ellipticgaussian.h"
#include "WaveFunctions/repulsiveellipticgaussian.h"
#include "WaveFunctions/nn_envelope.h"
#include "InitialStates/initialstate.h"
#include "Math/random.h"
#include "Solvers/metropolishastings.h"


VMCOptimizer_NN::VMCOptimizer_NN(
    unsigned int numberOfDimensions,
    unsigned int numberOfParticles,
    unsigned int numberOfEquilibrationSteps,
    double timeStep,
    std::unique_ptr<Hamiltonian> hamiltonian,
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
    double Adam_ktol,
    std::ofstream* logfile,
    std::ofstream* outfile,
    std::ofstream* paramsfile
) : m_numberOfDimensions(numberOfDimensions),
m_numberOfParticles(numberOfParticles),
m_numberOfEquilibrationSteps(numberOfEquilibrationSteps),
m_timeStep(timeStep),
m_hamiltonian(std::move(hamiltonian)),
m_solverFactory(solverFactory),
m_seed(seed),
m_Nhid(Nhid),
m_helpDecay(helpDecay),
m_optEquil(optEquil),
m_nSamples(nSamples),
m_nPretrainSteps(nPretrainSteps),
m_nEnergySteps(nEnergySteps),
m_nAdiabSteps(nAdiabSteps),
m_maxStrength(maxStrength),
m_lr(lr),
m_Adam_ktol(Adam_ktol),
m_logfile(logfile),
m_outfile(outfile),
m_paramsfile(paramsfile) {}

std::vector<double> VMCOptimizer_NN::optimize(std::unique_ptr<WaveFunction> wf_train) {
    auto rng = std::make_unique<Random>(m_seed == 0
        ? std::chrono::system_clock::now().time_since_epoch().count()
        : m_seed);
    auto wf_nn = std::make_unique<NN_envelope>(
        m_numberOfParticles,
        m_numberOfDimensions,
        m_numberOfParticles * m_numberOfDimensions,
        m_Nhid,
        m_helpDecay
    );
    NeuralNetwork* nnptr = &(wf_nn->net());
    torch::optim::Adam optimizer(
        nnptr->parameters(),
        torch::optim::AdamOptions(m_lr).betas({ 0.9, 0.999 }).eps(1e-6)
    );
    m_hamiltonian->set_percStrength(0);
    auto system = std::make_unique<System>(
        std::move(m_hamiltonian),
        std::move(wf_train)
    );
    std::cout << "Created NN system\n";
    
    system->setParticles(
        setupRandomUniformInitialState(m_numberOfDimensions, m_numberOfParticles, *rng)
    );
    system->setSolver(m_solverFactory(std::move(rng)));
    system->runEquilibrationSteps(m_timeStep, m_optEquil);
    // save positions
    std::vector<std::vector<double>> saved_pos = std::vector<std::vector<double>>(m_numberOfParticles);
    for (unsigned i = 0; i < m_numberOfParticles; i++)
    for (unsigned j = 0; j < m_numberOfDimensions; j++)
    saved_pos[i].push_back(system->getParticles()[i]->getPosition()[j]);
    auto tempsampler = system->runMetropolisSteps(m_timeStep, m_nSamples * 10);
    std::cout << "DEBUG: tempsampler energy = " << tempsampler->getEnergy() << " +- " << tempsampler->getError() << std::endl;
    wf_train = std::move(system->setWaveFunction(std::move(wf_nn)));
    // -------- PRE-TRAINING --------
    // print log header
    if (m_logfile) {
        *m_logfile << "# PRETRAINING\n#";
        const unsigned int width = 20;
        *m_logfile << std::setw(width-1) << "K,"
            << std::setw(width) << "elapsed time,"
            << std::setw(width) << "acceptance ratio " << std::endl;
    }
    std::cout << "Running Adam optimization without interactions...\n";
    *m_outfile << "# Running Adam optimization without interactions...\n";
    for (int step = 0; step < m_nPretrainSteps; step++) {
        std::cout << "\rPretrain step #" << step << std::flush;
        // if (step % 5 == 0) {
            // system->runEquilibrationSteps(m_timeStep, m_optEquil);
            // auto tempsampler = system->runMetropolisSteps_NN(m_timeStep, m_nSamples, *wf_train);
            // std::cout << "DEBUG: step " << step << " tempsampler energy = " << tempsampler->getEnergy() << std::endl;
            // resetPos(system, )
        // }
        // system->runEquilibrationSteps(m_timeStep, 50);

        std::unique_ptr<NNsampler> sampler =
            system->runMetropolisSteps_NN_pretrain(m_timeStep, m_nSamples, *wf_train);
        if (m_logfile) sampler->logOutput(*m_logfile);
        std::vector<double> dKdW = sampler->get_dKdW();
        // Adam minimizes, but we want to maximize K
        for (double& g : dKdW) {
            g = -g;
        }
        optimizer.zero_grad();
        nnptr->setGrads(dKdW);
        torch::nn::utils::clip_grad_norm_(nnptr->parameters(), 10);   // for stability
        optimizer.step();
        if (sampler->get_K() > m_Adam_ktol) break;
    }
    std::cout << "\nPretraining complete.\n";

    // -------- TRAINING --------
    // PRINT new header
    if (m_logfile) {
        *m_logfile << "\n# TRAINING\n#";
        const unsigned int width = 20;
        *m_logfile << std::setw(width-1) << "energy," << std::setw(width) << "variance,"
            << std::setw(width) << "error," << std::setw(width) << "Hint strength,"
            << std::setw(width) << "elapsed time," << std::setw(width) << "acceptance ratio "
            << std::endl;
    }
    std::cout << "Running Adam optimization with interactions...\n";
    *m_outfile << "# Running Adam optimization with interactions...\n";
    system->runEquilibrationSteps(m_timeStep, m_optEquil);
    for (int step = 0; step < m_nEnergySteps; step++) {
        // double percStrength = step / (double)m_nEnergySteps;
        double percStrength = (double) step / (double) m_nAdiabSteps;
        if (percStrength > 1) percStrength = 1;
        system->getHamiltonian().set_percStrength(percStrength);
        std::cout << "\rTrain step #" << std::setw(log10(m_nEnergySteps) + 1) << step
            << "   strength = " << std::scientific
            << system->getHamiltonian().get_interaction_strength() << std::flush;

        system->runEquilibrationSteps(m_timeStep, 50);
        auto sampler = system->runMetropolisSteps(m_timeStep, m_nSamples);
        if (m_logfile) sampler->logOutput(*m_logfile, {system->getHamiltonian().get_interaction_strength()});
        auto dEdW = sampler->get_dEdW();
        optimizer.zero_grad();
        nnptr->setGrads(dEdW);
        torch::nn::utils::clip_grad_norm_(nnptr->parameters(), 10);
        optimizer.step();
    }
    std::cout << "\nTraining complete.\n";

    if (m_paramsfile) {
        for (auto p : nnptr->getParams()) *m_paramsfile << p << std::endl;
    }
    return nnptr->getParams();
}


// deprecated
void VMCOptimizer_NN::resetPos(std::vector<std::unique_ptr<Particle>>& particles, std::vector<std::vector<double>>& pos) {
    for (unsigned i = 0; i < m_numberOfParticles; i++)
        for (unsigned j = 0; j < m_numberOfDimensions; j++)
            particles[i]->setPosition(pos[i][j], j);
}