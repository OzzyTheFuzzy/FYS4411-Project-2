// make clean
// ./compile_project

// Examples:
//  ./vmc bf 100 3 --alpha 0.50
//  ./vmc is 0.005 100 3 --alpha 0.50
//
// Example running alpha optimization with BFGS:
//  ./vmc bf 100 3 --opt bfgs
//  ./vmc is 0.005 100 3 --opt bfgs
//
// Example running parallel chains:
//  export OMP_NUM_THREADS=4
//  ./vmc is 0.02 500 3 --alpha 0.5 --replicas 4
//
// Example running interacting case:
//  ./vmc bf 10 3 --interact --a 0.0043 --gamma 2.82843 --beta 2.82843 --alpha 0.48 --replicas 4
//
// Example to use RBM:
//  ./vmc bf 2 2 --rbm
//  ./vmc is 0.02 2 2 --rbm
//  ./vmc is 0.02 10 3 --interact --rbm

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>
#include <limits>
#include <sstream>
#include <cmath>

#include "system.h"
#include "WaveFunctions/anisotropicgaussian.h"
#include "WaveFunctions/simplegaussian.h"
#include "WaveFunctions/boltzmannmachine.h"
#include "WaveFunctions/nn_envelope.h"
#include "Hamiltonians/elliptictrap.h"
#include "Hamiltonians/repulsiveho.h"
#include "InitialStates/initialstate.h"
#include "Solvers/metropolis.h"
#include "Solvers/importancesampling.h"
#include "Math/random.h"
#include "particle.h"
#include "sampler.h"
#include "RBMsampler.h"
#include "NNsampler.h"
#include "Solvers/montecarlo.h"
#include "optimizer.h"
#include "utilities.h"
#include "common.h"

#include <torch/torch.h>

int main(int argc, char** argv) {

    // Testing the "torch" library
    //torch::Tensor tensor = torch::eye(3);
    //std::cout << tensor << std::endl;
    //return 0;

    // ---------------- Defaults ----------------
    const int baseSeed = 2023;

    unsigned int N = 10;
    unsigned int D = 3;

    const double omega = 1.0;

    // bf: spatial stepLength, is: time step dt
    double stepLength = 0.5;
    double dt = 0.005;

    Mode mode = Mode::Importance;

    // alpha settings
    const double alphaMin = 0.2;
    const double alphaMax = 0.8;

    // optimization steps (can be same order as scan for stability)
    const unsigned int optEquil = 10000;
    const unsigned int optSteps = 100000;

    // production steps 
    const unsigned int prodEquil = static_cast<unsigned int>(50000); //1e5
    const unsigned int prodSteps = static_cast<unsigned int>(524288); //=2^{19} else 1e6

    // alpha selection mode
    double userAlpha = 0.5;
    short useOptimization = 0;
    std::string optMethod = "";

    // Interaction case
    bool useInteraction = false;
    double hardCoreA = 0.0043;
    double gamma = 1.0;
    double beta = 1.0;

    // Restricted Boltzman Machine
    bool useRBM = false;
    unsigned int Nh = 2;

    // Density calculation
    bool useDensityMode = false;
    bool densityOnly = false;
    bool useNoJastrow = false;
    unsigned int densityBins = 200;
    double densityZMax = 4.0;
    double densityRhoMax = 4.0;

    // Output folders 
    const std::string txtDir = "data/.txt/";

    // Number of replicas with no paralelization by defaults
    int nReplicas = 1;

    // NN parameters
    bool useNN = false;
    const int Nhid = 32;
    const double lr = 1e-3;
    const int nPretrainSteps = 5000;   // maximize K
    const int nEnergySteps = 40000;  // minimize E
    const int nSamples = 10000;  // Metropolis steps per update
    const double initRange = 1.0;   // for particle setup
    const double strengthRate = 20;   // hardcore potential strength increase per step
    auto wf_train = AnisotropicGaussian(0.5, beta); // Non-interacting training target

    // ---------------------Parse CLI---------------------
    // First parse bf/is, N, D
    if (argc > 1) {
        std::string m = argv[1];

        if (m == "is") {
            mode = Mode::Importance;

            if (argc < 3) {
                std::cerr << "Usage: ./vmc is <dt> [N D] [--alpha x]\n";
                return 1;
            }

            dt = std::stod(argv[2]);

            int idx = 3;

            if (idx < argc && std::string(argv[idx]).rfind("--", 0) != 0) {
                N = static_cast<unsigned int>(std::stoul(argv[idx++]));
            }
            if (idx < argc && std::string(argv[idx]).rfind("--", 0) != 0) {
                D = static_cast<unsigned int>(std::stoul(argv[idx++]));
            }
        }
        else if (m == "bf") {
            mode = Mode::BruteForce;

            int idx = 2;

            if (idx < argc && std::string(argv[idx]).rfind("--", 0) != 0) {
                N = static_cast<unsigned int>(std::stoul(argv[idx++]));
            }
            if (idx < argc && std::string(argv[idx]).rfind("--", 0) != 0) {
                D = static_cast<unsigned int>(std::stoul(argv[idx++]));
            }
        }
        else {
            std::cerr << "Error: first argument must be 'bf' or 'is'.\n";
            return 1;
        }
    }

    // Parse optional flags anywhere
    for (int i = 1; i < argc; i++) {
        std::string tok = argv[i];

        if (tok == "--alpha") {userAlpha = std::stod(argv[i + 1]);}
        if (tok == "--opt") {
            optMethod = argv[i + 1];
            if (optMethod == "bfgs") {useOptimization = 1;}
            else if (optMethod == "doubleAdam") {useOptimization = 2;}
            else {
                std::cerr << "Unknown optimization method: " << optMethod << "\n";
                return 1;
            }
        }
        if (tok == "--rbm") {useRBM = true;}
        if (tok == "--NN") {useNN = true;}
        if (tok == "--replicas") {
            nReplicas = std::stoi(argv[i + 1]);
            if (nReplicas < 1) {
                std::cerr << "Error: --replicas must be >= 1\n";
                return 1;
            }
        }
        if (tok == "--interact") {useInteraction = true;}
        if (tok == "--a") {hardCoreA = std::stod(argv[i + 1]);}
        if (tok == "--gamma") {gamma = std::stod(argv[i + 1]);}
        if (tok == "--beta") {beta = std::stod(argv[i + 1]);}
        if (tok == "--density") {useDensityMode = true;}
        if (tok == "--density-only") {
            useDensityMode = true;
            densityOnly = true;
        }
        if (tok == "--nojastrow") {useNoJastrow = true;}
        if (tok == "--bins") {densityBins = static_cast<unsigned int>(std::stoul(argv[i + 1]));}
        if (tok == "--zmax") {densityZMax = std::stod(argv[i + 1]);}
        if (tok == "--rhomax") {densityRhoMax = std::stod(argv[i + 1]);}
    }

    // RBM parameters must be initialized after N, D, and Nh are known.
    std::vector<std::vector<double>> a(N,std::vector<double>(D, 0.0));
    std::vector<double> b(Nh,0.0);
    std::vector<std::vector<std::vector<double>>> W(N,std::vector<std::vector<double>>(D,std::vector<double>(Nh, 0.01)));

    const double stepParam = (mode == Mode::Importance) ? dt : stepLength;

    std::cout << "\n=== Mode: " << modeToString(mode)
        << " | Local energy: analytic"
        << " | N=" << N << " D=" << D << " gamma=" << gamma <<" ===\n";
    if (mode == Mode::Importance) {
        std::cout << "Using dt = " << dt << "\n";
    }
    else {
        std::cout << "Using stepLength = " << stepLength << "\n";
    }

    // ---------------- Determine alpha ----------------
    double bestAlpha = userAlpha;
    double bestEnergy = std::numeric_limits<double>::quiet_NaN();

    const std::string modelTag = useInteraction ? "_interac" : "_";

    if (useOptimization == 1) {
        std::cout << "Running BFGS-style optimization for alpha...\n";

        auto objective = [&](double alpha) -> ObjectiveResult {
            RunResult r = runVMC(N, D, optSteps, optEquil, omega, alpha, stepParam, mode, baseSeed + static_cast<int>(1000 * alpha), 
                false, false, useInteraction, useRBM, a, b, W, beta, gamma, hardCoreA);
            ObjectiveResult out;
            out.energy = r.energy;
            out.gradient = r.gradient;
            return out;
            };

        std::string optFile = txtDir + "opt_path" + modelTag + "_mode_" + modeToString(mode);
        if (mode == Mode::Importance) optFile += "_dt_" + doubleToTag(dt);
        optFile += "_N" + std::to_string(N) + "_D" + std::to_string(D) + ".txt";

        OptimizationResult opt = optimizeAlphaBFGS1D(
            objective,
            userAlpha, // initial alpha guess
            alphaMin,
            alphaMax,
            20, // max iterations
            1e-4, // tolerance on |dE/dalpha|
            true, // print to terminal
            optFile // save optimizer path to file
        );

        std::cout << "Saved optimizer path to: " << optFile << "\n";

        bestAlpha = opt.alpha_opt;
        bestEnergy = opt.energy_opt;

        std::cout << "Optimization finished after " << opt.iterations << " iterations\n";
        std::cout << "Optimal alpha = " << bestAlpha
            << ", energy = " << bestEnergy
            << ", gradient = " << opt.gradient_opt
            << ", converged = " << opt.converged << "\n";

    }
    else if (useOptimization == 2) {
        int seedNN = std::chrono::system_clock::now().time_since_epoch().count();
        auto rng = std::make_unique<Random>(seedNN);
        std::unique_ptr<MonteCarlo> solver;
        if (mode == Mode::Importance) {
            solver = std::make_unique<ImportanceSampling>(std::move(rng), 0.5);
        }
        else {
            solver = std::make_unique<Metropolis>(std::move(rng));
        }
        std::vector<std::unique_ptr<Particle>> particles;

        auto wf_nn = std::make_unique<NN_envelope>(N, D, N, Nhid);
        torch::optim::Adam optimizer(
            wf_nn->net().parameters(),
            torch::optim::AdamOptions(lr).betas({ 0.9, 0.999 }).eps(1e-8)
        );

        // -------- PRE-TRAINING --------
        std::cout << "Running Adam optimization without interactions...\n";

        auto hamiltonian = std::make_unique<RepulsiveHO>(omega, gamma * omega, 0);
        
        if (useInteraction) {
            particles = setupRandomUniformInitialStateNoOverlap(
                initRange, D, N, hardCoreA, *rng
            );
        }
        else {
            particles = setupRandomUniformInitialState(
                initRange, D, N, *rng
            );
        }
        auto system_pretrain = std::make_unique<System>(
            std::move(hamiltonian),
            std::move(wf_nn),
            std::move(solver),
            std::move(particles)
        );

        for (int step = 0; step < nPretrainSteps; step++) {
            system_pretrain->runEquilibrationSteps(stepParam, optEquil);
            auto sampler_pretrain = system_pretrain->runMetropolisSteps_NN(stepParam, nSamples, wf_train);
            auto dKdW = sampler_pretrain->get_dKdW();
            // Negate: Adam minimizes, but we want to maximize K
            for (auto& g : dKdW) g = -g;

            optimizer.zero_grad();
            setNetworkGrads(wf_nn->net(), dKdW);
            optimizer.step();

            if (step % 500 == 0)
                std::cout << "  step " << step
                << "  K = " << sampler_pretrain->get_K() << "\n";
        }

        // -------- TRAINING --------
        std::cout << "Running Adam optimization with interactions...\n";

        hamiltonian = std::make_unique<RepulsiveHO>(omega, gamma * omega, hardCoreA);
        if (useInteraction) {
            particles = setupRandomUniformInitialStateNoOverlap(
                initRange, D, N, hardCoreA, *rng
            );
        }
        else {
            particles = setupRandomUniformInitialState(
                initRange, D, N, *rng
            );
        }
        auto system = std::make_unique<System>(
            std::move(hamiltonian),
            std::move(wf_nn),
            std::move(solver),
            std::move(particles)
        );

        for (int step = 0; step < nEnergySteps; step++) {
            hamiltonian->set_hardcore_strength(step* strengthRate);
            
            system->runEquilibrationSteps(stepParam, optEquil);
            auto sampler = system->runMetropolisSteps_NN(stepParam, nSamples, wf_train);
            auto dEdW = sampler->get_dEdW();

            optimizer.zero_grad();
            setNetworkGrads(wf_nn->net(), dEdW);
            optimizer.step();

            if (step % 500 == 0)
                std::cout << "  step " << step
                << "  E = " << sampler->getEnergy() << "\n";
        }

    }
    else {
        std::cout << "Using alpha = " << bestAlpha << "\n";
    }

    // ---------------- Density calculation ----------------

    if (useDensityMode) {

        std::cout << "\n=== Density run with alpha=" << bestAlpha
            << " | " << (useNoJastrow ? "without" : "with") << " Jastrow"
            << " | bins=" << densityBins << " ===\n";

        RunResult dens = runVMC(N, D, prodSteps, prodEquil, omega, bestAlpha, stepParam, mode, baseSeed + 7777, 
            false, false,  useInteraction, useRBM, a, b, W, beta, gamma, hardCoreA, 
            densityOnly, true, useNoJastrow, densityBins, densityZMax, densityRhoMax);

        const std::string jTag = useNoJastrow ? "_noJ" : "_withJ";

        std::string densZFile = txtDir + "density_z" + modelTag + jTag
            + "_mode_" + modeToString(mode)
            + "_N" + std::to_string(N) + "_D" + std::to_string(D) + ".txt";

        std::string densRhoFile = txtDir + "density_rho" + modelTag + jTag
            + "_mode_" + modeToString(mode)
            + "_N" + std::to_string(N) + "_D" + std::to_string(D) + ".txt";

        std::ofstream outZ(densZFile);
        outZ << "# z density\n";
        outZ << std::setprecision(12);
        for (std::size_t i = 0; i < dens.densityZ.size(); ++i) {
            outZ << dens.densityZCenters[i] << " " << dens.densityZ[i] << "\n";
        }
        outZ.close();

        std::ofstream outRho(densRhoFile);
        outRho << "# rho density\n";
        outRho << std::setprecision(12);
        for (std::size_t i = 0; i < dens.densityRho.size(); ++i) {
            outRho << dens.densityRhoCenters[i] << " " << dens.densityRho[i] << "\n";
        }
        outRho.close();

        std::cout << "Saved z-density to: " << densZFile << "\n";
        std::cout << "Saved rho-density to: " << densRhoFile << "\n";

        return 0;
    }

    // ---------------- Production run ----------------

    std::string prodFile = txtDir + "energy" + modelTag + "mode_" + modeToString(mode);
    if (useRBM) prodFile += "_RBM_hidden_" + std::to_string(Nh);
    if (mode == Mode::Importance) prodFile += "_dt_" + doubleToTag(dt);
    if (gamma != 1.0) prodFile += "_gamma_" + doubleToTag(gamma);
    if (useInteraction) prodFile += "_inter";
    prodFile += "_N" + std::to_string(N) + "_D" + std::to_string(D) + ".txt";

    std::string histFile = txtDir + "energy_history" + modelTag + "_mode_" + modeToString(mode);
    if (mode == Mode::Importance) histFile += "_dt_" + doubleToTag(dt);
    histFile += "_N" + std::to_string(N) + "_D" + std::to_string(D) + ".txt";

    std::string gradientFile = txtDir + "gradients" + modelTag + "mode_" + modeToString(mode);
    if (useRBM) gradientFile += "_RBM_hidden" + std::to_string(Nh);
    if (mode == Mode::Importance) gradientFile += "_dt_" + doubleToTag(dt);
    if (gamma != 1.0) gradientFile += "_gamma_" + doubleToTag(gamma);
    if (useInteraction) gradientFile += "_inter";
    gradientFile += "_N" + std::to_string(N) + "_D" + std::to_string(D) + ".txt";

    std::cout << "\n=== Production run with alpha=" << bestAlpha
        << " | local energy eval: analytic" 
        << " | replicas=" << nReplicas
        << " ===\n";

    /*
    ParallelRunResult prodPar = runVMCReplicasParallel(
        N, D,
        prodSteps,
        prodEquil,
        omega,
        bestAlpha,
        stepParam,
        mode,
        baseSeed + 9999,
        nReplicas,
        true,
        useInteraction,
        useRBM,
        a,
        b,
        W,
        beta,
        gamma,
        hardCoreA
    );
    */

    RunResult result = runVMC(
        N, D,
        prodSteps,
        prodEquil,
        omega,
        bestAlpha,
        stepParam,
        mode,
        baseSeed + 7777,
        false,
        false,
        useInteraction,
        useRBM,
        a,
        b,
        W,
        beta,
        gamma,
        hardCoreA,
        densityOnly,
        true,
        useNoJastrow,
        densityBins,
        densityZMax,
        densityRhoMax
    );

    std::ofstream outProd(prodFile);
    if (!outProd) {
        std::cerr << "Error: could not open " << prodFile << "\n";
        return 1;
    }

    if (useRBM) {
        std::cout << "\n\nDEBUG: last if detected\n\n";
        std::ofstream outGrad(gradientFile);
        outGrad << "Gradients\n";
        outGrad << "=========\n\n";
        // Write a
        outGrad << "Gradient_a (" << result.gradienta.size() << " x ";
        if (!result.gradienta.empty()) outGrad << result.gradienta[0].size();
        else outGrad << 0;
        outGrad << ")\n";
        for (size_t i = 0; i < result.gradienta.size(); ++i) {
            outGrad << "Gradient_a[" << i << "] = ";
            for (size_t d = 0; d < result.gradienta[i].size(); ++d) {
                outGrad << result.gradienta[i][d];
                if (d + 1 < result.gradienta[i].size()) outGrad << " ";
            }
            outGrad << "\n";
        }
        outGrad << "\n";
        // Write b
        outGrad << "Gradient_b (" << b.size() << ")\n";
        outGrad << "Gradient_b = ";
        for (size_t j = 0; j < result.gradientb.size(); ++j) {
            outGrad << result.gradientb[j];
            if (j + 1 < result.gradientb.size()) outGrad << " ";
        }
        outGrad << "\n\n";
        // Write W
        outGrad << "Gradient_W (" << result.gradientw.size() << " x ";
        if (!result.gradientw.empty()) outGrad << result.gradientw[0].size();
        else outGrad << 0;
        outGrad << " x ";
        if (!result.gradientw.empty() && !result.gradientw[0].empty()) outGrad << result.gradientw[0][0].size();
        else outGrad << 0;
        outGrad << ")\n";
        for (size_t i = 0; i < result.gradientw.size(); ++i) {
            outGrad << "Gradient_W[" << i << "] =\n";
            for (size_t d = 0; d < result.gradientw[i].size(); ++d) {
                outGrad << "  ";
                for (size_t j = 0; j < result.gradientw[i][d].size(); ++j) {
                    outGrad << result.gradientw[i][d][j];
                    if (j + 1 < result.gradientw[i][d].size()) outGrad << " ";
                }
                outGrad << "\n";
            }
            outGrad << "\n";
        }
        outGrad.close();

        outProd << "# N D mode stepParam replicas energy acceptance cpu_seconds\n";
        outProd << N << " " << D << " " << modeToString(mode) << " "
            << stepParam << " "
            << nReplicas << " "
            << std::setprecision(12)
            << result.energy << " "
            << result.acceptance << " "
            << result.cpu_seconds << "\n";

    }
    /*
    else {
        outProd << "# N D mode stepParam omega alpha replicas energy_mean gradient_mean acceptance_mean energy_std energy_stderr wall_seconds mean_replica_cpu_seconds\n";
        outProd << N << " " << D << " " << modeToString(mode) << " "
            << stepParam << " " << omega << " " << bestAlpha << " "
            << nReplicas << " "
            << std::setprecision(12)
            << prodPar.mean.energy << " "
            << prodPar.mean.gradient << " "
            << prodPar.mean.acceptance << " "
            << prodPar.energy_std << " "
            << prodPar.energy_stderr << " "
            << prodPar.wall_seconds << " "
            << prodPar.mean.cpu_seconds << "\n";
    }
    */

    outProd.close();

    std::cout << "Saved production result to: " << prodFile << "\n";
    std::cout << "Saved gradient result to: " << gradientFile << "\n";

    /*
    std::ofstream outHist(histFile);
    if (!outHist) {
        std::cerr << "Error: could not open " << histFile << "\n";
        return 1;
    }

    outHist << "# replica step local_energy\n";
    outHist << std::setprecision(12);

    for (int r = 0; r < nReplicas; ++r) {
        const auto& hist = prodPar.replicas[r].energyHistory;
        for (std::size_t i = 0; i < hist.size(); ++i) {
            outHist << r << " " << (i + 1) << " " << hist[i] << "\n";
        }
    }

    outHist.close();
    std::cout << "Saved energy histories to: " << histFile << "\n";
    */
    return 0;
}