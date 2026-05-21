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
    const runConfig& cfg,
    std::unique_ptr<Hamiltonian> hamiltonian,
    SolverFactory solverFactory,
    ActivationFunc actFun,
    std::ofstream* logfile,
    std::ofstream* outfile,
    std::ofstream* paramsfile
) : 
    m_cfg(cfg),
    m_hamiltonian(std::move(hamiltonian)),
    m_solverFactory(solverFactory),
    m_actFun(actFun),
    m_logfile(logfile),
    m_outfile(outfile),
    m_paramsfile(paramsfile) 
{}

std::vector<double> VMCOptimizer_NN::optimize(std::unique_ptr<WaveFunction> wf_train) {
    auto rng = std::make_unique<Random>(m_cfg.seed == 0
        ? std::chrono::system_clock::now().time_since_epoch().count()
        : m_cfg.seed);
    auto wf_nn = std::make_unique<NN_envelope>(
        m_cfg.numberOfParticles,
        m_cfg.numberOfDimensions,
        m_cfg.numberOfParticles * m_cfg.numberOfDimensions,
        m_cfg.Nhid,
        m_cfg.helpDecay,
        m_actFun
    );
    NeuralNetwork* nnptr = &(wf_nn->net());
    torch::optim::Adam optimizer(
        nnptr->parameters(),
        torch::optim::AdamOptions(m_cfg.lr).betas({ 0.9, 0.999 }).eps(1e-6)
    );
    m_hamiltonian->set_percStrength(0);
    auto system = std::make_unique<System>(
        std::move(m_hamiltonian),
        std::move(wf_train)
    );
    std::cout << "Created NN system\n";

    system->setParticles(
        setupRandomUniformInitialState(m_cfg.numberOfDimensions, m_cfg.numberOfParticles, *rng)
    );
    system->setSolver(m_solverFactory(std::move(rng)));
    system->runEquilibrationSteps(m_cfg.timeStep, m_cfg.equilibrationSteps);
    // save positions
    // std::vector<std::vector<double>> saved_pos = std::vector<std::vector<double>>(m_numberOfParticles);
    // for (unsigned i = 0; i < m_numberOfParticles; i++)
    //     for (unsigned j = 0; j < m_numberOfDimensions; j++)
    //         saved_pos[i].push_back(system->getParticles()[i]->getPosition()[j]);
    auto tempsampler = system->runMetropolisSteps(m_cfg.timeStep, m_cfg.metropolisSteps * 10); // trying out non interactive analytical wavefunction
    std::cout << "DEBUG: tempsampler energy = " << tempsampler->getEnergy() << " +- " << tempsampler->getError() << std::endl;
    wf_train = std::move(system->setWaveFunction(std::move(wf_nn)));    // swap wavefunctions
    // -------- PRE-TRAINING --------
    // print log header
    if (m_logfile) {
        *m_logfile << "# PRETRAINING\n#";
        const unsigned int width = 20;
        *m_logfile << std::setw(width - 1) << "K,"
            << std::setw(width) << "elapsed time,"
            << std::setw(width) << "acceptance ratio " << std::endl;
    }
    std::cout << "Running Adam optimization without interactions...\n";
    *m_outfile << "# Running Adam optimization without interactions...\n";

    double best_val = 0, min_improvement = m_cfg.min_improvement;
    int patience_counter = 0, times_reached_plateau = 0;
    for (int step = 0; step < m_cfg.nPretrainSteps; step++) {
        std::cout << "\rPretrain step #" << step << std::flush;

        std::unique_ptr<NNsampler> sampler =
            system->runMetropolisSteps_NN_pretrain(m_cfg.timeStep, m_cfg.metropolisSteps, *wf_train);
        if (m_logfile) sampler->logOutput(*m_logfile);
        std::vector<double> dKdW = sampler->get_dKdW();
        double K = sampler->get_K();
        if (K > m_cfg.Adam_ktol) break;
        if (checkPlateau(K, best_val, patience_counter, min_improvement, true)) {
            min_improvement *= 0.1;
            double new_lr = m_cfg.lr * pow(10, -(++times_reached_plateau));
            set_lr(optimizer, new_lr);
            std::cout << "\t Plateau detected at best_K = " << best_val
                << ". Setting base lr = " << new_lr << std::endl;
        }
        // proceed with optimization, but Adam minimizes, instead we want to maximize K
        for (double& g : dKdW) {
            g = -g;
        }
        optimizer.zero_grad();
        nnptr->setGrads(dKdW);
        torch::nn::utils::clip_grad_norm_(nnptr->parameters(), 10);   // for stability
        optimizer.step();
    }
    std::cout << "\nPretraining complete.\n";

    // -------- TRAINING --------
    // PRINT new header
    if (m_logfile) {
        *m_logfile << "\n# TRAINING\n#";
        const unsigned int width = 20;
        *m_logfile << std::setw(width - 1) << "energy," << std::setw(width) << "variance,"
        << std::setw(width) << "error," << std::setw(width) << "Hint strength,"
        << std::setw(width) << "elapsed time," << std::setw(width) << "acceptance ratio "
        << std::endl;
    }
    std::cout << "Running Adam optimization with interactions...\n";
    *m_outfile << "# Running Adam optimization with interactions...\n";
    set_lr(optimizer, m_cfg.lr);
    patience_counter = 0; times_reached_plateau = 0;
    min_improvement = m_cfg.min_improvement; best_val = 0;
    system->runEquilibrationSteps(m_cfg.timeStep, m_cfg.equilibrationSteps);
    for (int step = 0; step < m_cfg.nEnergySteps; step++) {
        // double percStrength = step / (double)m_nEnergySteps;
        if (step <= m_cfg.nAdiabSteps) {
            double percStrength = (double)step / (double)m_cfg.nAdiabSteps;
            system->getHamiltonian().set_percStrength(percStrength);
        }
        std::cout << "\rTrain step #" << std::setw(log10(m_cfg.nEnergySteps) + 1) << step
            << "   strength = " << std::scientific
            << system->getHamiltonian().get_interaction_strength() << std::flush;

        system->runEquilibrationSteps(m_cfg.timeStep, 50);
        auto sampler = system->runMetropolisSteps(m_cfg.timeStep, m_cfg.metropolisSteps);
        if (m_logfile) sampler->logOutput(*m_logfile, { system->getHamiltonian().get_interaction_strength() });
        auto dEdW = sampler->get_dEdW();
        double E = sampler->getEnergy();
        if (step == m_cfg.nAdiabSteps) best_val = E;
        if (step > m_cfg.nAdiabSteps && checkPlateau(E, best_val, patience_counter, min_improvement, false)) {
            min_improvement *= 0.1;
            double new_lr = m_cfg.lr * pow(10, -(++times_reached_plateau));
            set_lr(optimizer, new_lr);
            std::cout << "\t Plateau detected at best_E = " << best_val
                << ". Setting base lr = " << new_lr << std::endl;
        }
        // proceed with optimization
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

void VMCOptimizer_NN::set_lr(torch::optim::Adam& optimizer, double new_lr) {
    for (auto& group : optimizer.param_groups()) {
        if (group.has_options()) {
            auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
            options.lr(new_lr);
        }
    }
}

bool VMCOptimizer_NN::checkPlateau(
    double current_val, double& best_val, int& patience_counter, double min_improvement, bool increase
) {
    if (increase && current_val > best_val + min_improvement) {
        // Significant improvement in K found
        best_val = current_val;
        patience_counter = 0;
        return false;
    }
    else if (!increase && current_val < best_val - min_improvement) {
        // Significant improvement in E found
        best_val = current_val;
        patience_counter = 0;
        return false;
    }
    else {
        // No significant improvement
        patience_counter++;
        if (patience_counter >= m_cfg.max_patience) {
            patience_counter = 0;
            return true;
        }
        return false;
    }
}

// deprecated
void VMCOptimizer_NN::resetPos(std::vector<std::unique_ptr<Particle>>& particles, std::vector<std::vector<double>>& pos) {
    for (unsigned i = 0; i < m_cfg.numberOfParticles; i++)
        for (unsigned j = 0; j < m_cfg.numberOfDimensions; j++)
            particles[i]->setPosition(pos[i][j], j);
}