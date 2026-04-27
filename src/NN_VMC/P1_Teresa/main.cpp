// make clean
// ./compile_project

// Examples runing an alpha scan:
//  ./vmc bf nu 100 3  (use brute force metropolis, with the optimal alpha use numericla Laplacian for E, N=100, D=3)
//  ./vmc bf an 100 3  (use brute force metropolis, with the optimal alpha use analytical expresion for F, N=100, D=3)
//  ./vmc bf 100 3     (same as bf an)
//  ./vmc is 0.005 an 100 3 (use importance sampling, dt 0.005, with the optimal alpha use analytical expresion for F, N=100, D=3)
//  ./vmc is 0.005 nu 100 3  -> error

// Examples with no alpha scan:
//  ./vmc bf nu 100 3 --alpha 0.50  (Here the alpha value selected is 0.5)
//  ./vmc is 0.005 an 100 3 --alpha 0.50

// Example runing alpha optimization with BFGS
//  ./vmc bf nu 100 3 --opt bfgs
//  ./vmc is 0.005 an 100 3 --opt bfgs

// Example runing parallel chains:
//  export OMP_NUM_THREADS=4
//  ./vmc is 0.02 an 500 3 --alpha 0.5 --replicas 4  (In this example we run 4 parallel chains)

// Example runing interacting case:
// ./vmc bf an 10 3 --interact --a 0.0043 --gamma 2.82843 --beta 2.82843 --alpha 0.48 --replicas 4 (This fixes alpha value and applies paralelization)
 
// Example to obtain one-body densities:
// ./vmc bf an 50 3 --interact --a 0.0043 --gamma 2.82843 --beta 2.82843 --alpha 0.490065 --density-only (with Jastrow factor)
// ./vmc bf an 50 3 --interact --a 0.0043 --gamma 2.82843 --beta 2.82843 --alpha 0.490065 --density-only --nojastrow  (with no jastrow factor)
// ./vmc bf an 50 3 --interact --a 0.0043 --gamma 2.82843 --beta 2.82843 --alpha 0.490065 --density-only --bins 250 --zmax 4.0 --rhomax 3.0 (setting histogram resolution)

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
#include "WaveFunctions/simplegaussian.h"
#include "WaveFunctions/boltzmannmachine.h"
#include "Hamiltonians/harmonicoscillator.h"
#include "InitialStates/initialstate.h"
#include "Solvers/metropolis.h"
#include "Solvers/importancesampling.h"
#include "Math/random.h"
#include "particle.h"
#include "sampler.h"
#include "RBMsampler.h"
#include "Solvers/montecarlo.h"
#include "optimizer.h"
#include "utilities.h"

#include <torch/torch.h>

static void testRBMMilestone1()
{
    const unsigned int N = 2;
    const unsigned int D = 2;
    const unsigned int Nh = 2;

    // Small RBM parameters
    std::vector<std::vector<double>> a(N, std::vector<double>(D, 0.0));
    std::vector<double> b(Nh, 0.0);
    std::vector<std::vector<std::vector<double>>> W(
        N,
        std::vector<std::vector<double>>(D, std::vector<double>(Nh, 0.01))
    );

    // Two particles in 2D
    std::vector<std::unique_ptr<Particle>> particles;
    particles.push_back(std::make_unique<Particle>(std::vector<double>{0.2, -0.1}));
    particles.push_back(std::make_unique<Particle>(std::vector<double>{-0.3, 0.4}));

    BoltzmannMachine rbm(a, b, W);
    HarmonicOscillator ho(1.0, false, 1e-3);

    const double psi = rbm.evaluate(particles);
    const double lapPsi = rbm.computeDoubleDerivative(particles);
    const double EL = ho.computeLocalEnergy(rbm, particles);

    std::cout << "\n--- RBM Milestone 1 test ---\n";
    std::cout << "Psi                = " << psi << "\n";
    std::cout << "Laplacian Psi      = " << lapPsi << "\n";
    std::cout << "Local energy       = " << EL << "\n";

    if (!std::isfinite(psi) || !std::isfinite(lapPsi) || !std::isfinite(EL)) {
        std::cerr << "RBM Milestone 1 FAILED: non-finite value detected.\n";
    } else {
        std::cout << "RBM Milestone 1 PASSED: all values are finite.\n";
    }
}

int main(int argc, char** argv) {

    // Testing some things with new RBM
    testRBMMilestone1();
    return 0;

    // Testing the "torch" library
    //torch::Tensor tensor = torch::eye(3);
    //std::cout << tensor << std::endl;
    //return 0;

    // ---------------- Defaults ----------------
    const int baseSeed = 2023;

    unsigned int N = 10;
    unsigned int D = 2;

    const double omega = 1.0;

    // bf: spatial stepLength, is: time step dt
    double stepLength = 0.5;
    double dt = 0.005;

    Mode mode = Mode::BruteForce;
    EvalType productionEval = EvalType::Analytic;

    // alpha scan settings
    const double alphaMin = 0.2;
    const double alphaMax = 0.8;
    const int nAlpha = 60;

    // cheap scan steps
    unsigned int scanEquil = 10000;
    unsigned int scanSteps = 100000;

    // production steps 
    const unsigned int prodEquil = static_cast<unsigned int>(50000); //1e5
    const unsigned int prodSteps = static_cast<unsigned int>(524288); //=2^{19} else 1e6

    // optimization steps (can be same order as scan for stability)
    const unsigned int optEquil = 10000;
    const unsigned int optSteps = 100000;

    // numerical Laplacian FD step (only used when productionEval == nu in bf mode)
    const double hFD = 1e-3;

    // alpha selection mode
    bool skipScan = false;
    double userAlpha = 0.5;
    bool useOptimization = false;
    std::string optMethod = "";

    // Interaction case
    bool useInteraction = false;
    double hardCoreA = 0.0043;
    double gamma = 2.82843;
    double beta = 2.82843;

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

    // ---------------------Parse CLI---------------------
    // First parse bf/is, eval type, N, D
    if (argc > 1) {
        std::string m = argv[1];

        if (m == "is") {
            mode = Mode::Importance;
            if (argc < 3) {
                std::cerr << "Usage: ./vmc is <dt> [an] [N D] [--alpha x]\n";
                return 1;
            }
            dt = std::stod(argv[2]);

            int idx = 3;
            if (idx < argc) {
                std::string token = argv[idx];
                if (token == "nu") {
                    std::cerr << "Error: numerical Laplacian option 'nu' is not allowed with importance sampling.\n";
                    return 1;
                } else if (token == "an") {
                    idx++;
                }
            }
            if (idx < argc && std::string(argv[idx]).rfind("--", 0) != 0) N = static_cast<unsigned int>(std::stoul(argv[idx++]));
            if (idx < argc && std::string(argv[idx]).rfind("--", 0) != 0) D = static_cast<unsigned int>(std::stoul(argv[idx++]));

            productionEval = EvalType::Analytic; // forced for is

        } else { // bf
            mode = Mode::BruteForce;

            int idx = 2;
            if (idx < argc) {
                std::string token = argv[idx];
                if (token == "an") { productionEval = EvalType::Analytic; idx++; }
                else if (token == "nu") { productionEval = EvalType::Numerical; idx++; }
            }
            if (idx < argc && std::string(argv[idx]).rfind("--", 0) != 0) N = static_cast<unsigned int>(std::stoul(argv[idx++]));
            if (idx < argc && std::string(argv[idx]).rfind("--", 0) != 0) D = static_cast<unsigned int>(std::stoul(argv[idx++]));
        }
    }

    // Parse optional flags anywhere
    for (int i = 1; i < argc; i++) {
        std::string tok = argv[i];

        if (tok == "--alpha") {
            userAlpha = std::stod(argv[i + 1]);
            skipScan = true;
        }
        if (tok == "--opt") {
            optMethod = argv[i + 1];
            if (optMethod == "bfgs") {
                useOptimization = true;
            } else {
                std::cerr << "Unknown optimization method: " << optMethod << "\n";
                return 1;
            }
        }
        if (tok == "--replicas") {
            nReplicas = std::stoi(argv[i+1]);
            if (nReplicas < 1) {
                std::cerr << "Error: --replicas must be >= 1\n";
                return 1;
            }
        }
        if (tok == "--interact") {
            useInteraction = true;
        }
        if (tok == "--a") {
            hardCoreA = std::stod(argv[i + 1]);
        }
        if (tok == "--gamma") {
            gamma = std::stod(argv[i + 1]);
        }
        if (tok == "--beta") {
            beta = std::stod(argv[i + 1]);
        }
        if (tok == "--density") {
            useDensityMode = true;
        }
        if (tok == "--density-only") {
            useDensityMode = true;
            densityOnly = true;
        }
        if (tok == "--nojastrow") {
            useNoJastrow = true;
        }
        if (tok == "--bins") {
            densityBins = static_cast<unsigned int>(std::stoul(argv[i + 1]));
        }
        if (tok == "--zmax") {
            densityZMax = std::stod(argv[i + 1]);
        }
        if (tok == "--rhomax") {
            densityRhoMax = std::stod(argv[i + 1]);
        }
    }

    const double stepParam = (mode == Mode::Importance) ? dt : stepLength;

    std::cout << "\n=== Mode: " << modeToString(mode)
              << " | Production eval: " << evalToString(productionEval)
              << " | N=" << N << " D=" << D << " ===\n";
    if (mode == Mode::Importance) {
        std::cout << "Using dt = " << dt << "\n";
    } else {
        std::cout << "Using stepLength = " << stepLength << "\n";
    }

    // ---------------- Determine alpha ----------------
    double bestAlpha = alphaMin;
    double bestEnergy = std::numeric_limits<double>::infinity();

    const std::string modelTag = useInteraction ? "_hs" : "_ideal";

    std::string scanFile = txtDir + "E_vs_alpha" + modelTag + "_mode_" + modeToString(mode);
    if (mode == Mode::Importance) scanFile += "_dt_" + doubleToTag(dt);
    scanFile += "_N" + std::to_string(N) + "_D" + std::to_string(D) + ".txt";
    
    if (skipScan) {
        bestAlpha = userAlpha;
        std::cout << "Skipping alpha scan. Using alpha = " << bestAlpha << "\n";

    } else if (useOptimization){
        std::cout << "Running BFGS-style optimization for alpha...\n";
        
        auto objective = [&](double alpha) -> ObjectiveResult {
            RunResult r = runVMC( N, D, optSteps, optEquil, omega, 
                                  alpha, stepParam, mode, 
                                  baseSeed + static_cast<int>(1000*alpha), 
                                  false, hFD, false, false, useInteraction, beta, gamma, hardCoreA);
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
            0.4, // initial alpha guess
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

    } else {
        auto res = runAlphaScan ( N, D, scanSteps, scanEquil, omega, alphaMin,
                                  alphaMax, nAlpha, stepParam, mode, baseSeed,
                                  hFD, scanFile, useInteraction, beta, gamma, hardCoreA);
        bestAlpha = res.first;
        bestEnergy = res.second;

        std::cout << "Saved scan data to: " << scanFile << "\n";
        std::cout << "Best alpha from scan: " << bestAlpha << "  (E ~ " << bestEnergy << ")\n";        
    }

    // ---------------- Density calculation ----------------

    if (useDensityMode) {
        if (!skipScan) {
            std::cerr << "Error: density mode requires --alpha <value>.\n";
            return 1;
        }

        std::cout << "\n=== Density run with alpha=" << bestAlpha
                << " | " << (useNoJastrow ? "without" : "with") << " Jastrow"
                << " | bins=" << densityBins << " ===\n";

        RunResult dens = runVMC(
            N, D,
            prodSteps,
            prodEquil,
            omega,
            bestAlpha,
            stepParam,
            mode,
            baseSeed + 7777,
            false,
            hFD,
            false,
            false,
            useInteraction,
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
    bool useNumericalLaplacian = false;
    if (mode == Mode::BruteForce && productionEval == EvalType::Numerical) {
        useNumericalLaplacian = true;
    }

    std::string prodFile = txtDir + "final_energy" + modelTag + "_mode_"+ modeToString(mode);
    if (mode == Mode::Importance) prodFile += "_dt_" + doubleToTag(dt);
    prodFile += "_eval_" + evalToString((mode == Mode::Importance) ? EvalType::Analytic : productionEval);
    prodFile += "_N" + std::to_string(N) + "_D" + std::to_string(D) + ".txt";

    std::string histFile = txtDir + "energy_history"  + modelTag + "_mode_" + modeToString(mode);
    if (mode == Mode::Importance) histFile += "_dt_" + doubleToTag(dt);
    histFile += "_eval_" + evalToString((mode == Mode::Importance) ? EvalType::Analytic : productionEval);
    histFile += "_N" + std::to_string(N) + "_D" + std::to_string(D) + ".txt";

    std::cout << "\n=== Production run with alpha=" << bestAlpha
              << " | local energy eval: " << (useNumericalLaplacian ? "numerical" : "analytic")
              << " | replicas=" << nReplicas
              << " ===\n";

    ParallelRunResult prodPar = runVMCReplicasParallel(
        N, D,
        prodSteps,
        prodEquil,
        omega,
        bestAlpha,
        stepParam,
        mode,
        baseSeed + 9999,
        useNumericalLaplacian,
        hFD,
        nReplicas,
        true,
        useInteraction,
        beta,
        gamma,
        hardCoreA
    );

    std::ofstream outProd(prodFile);
    if (!outProd) {
        std::cerr << "Error: could not open " << prodFile << "\n";
        return 1;
    }
    outProd << "# N D mode stepParam omega alpha eval replicas energy_mean gradient_mean acceptance_mean energy_std energy_stderr wall_seconds mean_replica_cpu_seconds\n";
    outProd << N << " " << D << " " << modeToString(mode) << " "
            << stepParam << " " << omega << " " << bestAlpha << " "
            << (useNumericalLaplacian ? "nu" : "an") << " "
            << nReplicas << " "
            << std::setprecision(12)
            << prodPar.mean.energy << " "
            << prodPar.mean.gradient << " "
            << prodPar.mean.acceptance << " "
            << prodPar.energy_std << " "
            << prodPar.energy_stderr << " "
            << prodPar.wall_seconds << " "
            << prodPar.mean.cpu_seconds << "\n";

    outProd.close();

    std::cout << "Saved production result to: " << prodFile << "\n";

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

    return 0;
}