// make clean
// ./compile_project

// Example running parallel chains:
//  export OMP_NUM_THREADS=4
//  ./vmc is 0.02 100 3 --replicas 4
//
// Example running interacting case:
//  ./vmc bf 0.8 10 3 --interact 
//
// Example to use RBM:
//  ./vmc bf 0.8 2 2 
//  ./vmc is 0.02 2 2 
//  ./vmc is 0.02 10 3 --interact 
//  ./vmc is 0.02 2 2 --interact --replicas 4 --opt rbmadam --lr 0.05 --Nh 2 --gamma 2.82843
//
//          choose between "bf" and "is" for Brute Force metropolis and Importance Sampling respectively
//          the number that follows "bf" or "is" is the step length or time step respectively
//          The following two numbers, for example 2 2, represent the number of particles and dimensions respectively
//          "--interact" takes the interaction between particles
//          "--replicas 4" uses parallel computing, in this example computes 4 replicas. Run: export OMP_NUM_THREADS=4
//          "--opt rbmadam" is for the adam optimizer in RBM
//          "--lr 0.05" is the learning rate for the optimizer
//          "--Nh 2" is the hidden layers number in RBM
//          "--gamma 2.82843" is for the elliptical trap 

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
#include <random>

#include "system.h"
#include "WaveFunctions/boltzmannmachine.h"
#include "Hamiltonians/elliptictrap.h"
#include "InitialStates/initialstate.h"
#include "Solvers/metropolis.h"
#include "Solvers/importancesampling.h"
#include "Math/random.h"
#include "particle.h"
#include "RBMsampler.h"
#include "Solvers/montecarlo.h"
#include "rbmoptimizer.h"
#include "utilities.h"

int main(int argc, char** argv) {

    // ---------------- Defaults ----------------
    const int baseSeed = 2023;

    unsigned int N = 10;
    unsigned int D = 3;

    // bf: spatial stepLength, is: time step dt
    double stepLength = 0.8;  
    double dt = 0.02;

    Mode mode = Mode::Importance;

    // optimization steps 
    const unsigned int optEquil = 10000;
    const unsigned int optSteps = 100000;

    // production steps 
    const unsigned int prodEquil = static_cast<unsigned int>(50000); //1e5
    const unsigned int prodSteps = static_cast<unsigned int>(524288); //=2^{19} else 1e6

    // Interaction case
    bool useInteraction = false;

    // Elliptical trap
    double gamma = 1.0;

    // Restricted Boltzman Machine
    unsigned int Nh = 4;

    // RBM Adam optimization parameters
    bool useOptimization = false;
    std::string optMethod = "";
    int rbmAdamIterations = 100;
    double rbmLearningRate = 0.05;
    double rbmBeta1 = 0.9;
    double rbmBeta2 = 0.999;
    double rbmAdamEps = 1e-8;

    // Output folders 
    const std::string txtDir = "data/.txt/";

    // Number of replicas with no paralelization by defaults
    int nReplicas = 1;

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
            if (argc < 3) {
                std::cerr << "Usage: ./vmc bf <stepLength> [N D] [--alpha x]\n";
                return 1;
            }
            stepLength = std::stod(argv[2]);
            int idx = 3;
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

        if (tok == "--opt") {
            optMethod = argv[i + 1];
            if (optMethod == "rbmadam") {
                useOptimization = true;}
            else {
                std::cerr << "Unknown optimization method, write: rbmadam instead of" << optMethod << "\n";
                return 1;
            }
        }
        if (tok == "--lr") {
            rbmLearningRate = std::stod(argv[i + 1]);
        }
        if (tok == "--Nh") {
            Nh = std::stod(argv[i + 1]);
        }
        if (tok == "--replicas") {
            nReplicas = std::stoi(argv[i + 1]);
            if (nReplicas < 1) {
                std::cerr << "Error: --replicas must be >= 1\n";
                return 1;
            }
        }
        if (tok == "--interact") {useInteraction = true;}
        if (tok == "--gamma") {gamma = std::stod(argv[i + 1]);}
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

    // ---------------- Determine optimal parameters ----------------

    const std::string modelTag = useInteraction ? "_interac_" : "_";

    if (useOptimization) {
        std::cout << "\n=== RBM Adam optimization ===\n";
        std::cout << "Iterations = " << rbmAdamIterations
              << ", learning rate = " << rbmLearningRate << "\n";

        RBMAdamOptimizer rbmOptimizer(a, b, W, rbmLearningRate, rbmBeta1, rbmBeta2, rbmAdamEps);

        std::string OptFile = txtDir + "optimization" + modelTag + modeToString(mode);
        if (mode == Mode::Importance) OptFile += "_dt_" + doubleToTag(dt);
        if (mode == Mode::BruteForce) OptFile += "_sl_" + doubleToTag(stepLength);
        OptFile += "_lr_" + doubleToTag(rbmLearningRate)+ "_Nh_" + std::to_string(Nh);
        if (gamma != 1.0) OptFile += "_gamma_" + doubleToTag(gamma);
        OptFile += "_N" + std::to_string(N) + "_D" + std::to_string(D) + ".txt";

        std::ofstream outOpt(OptFile);
        outOpt << "# iter energy acceptance cpu_seconds\n";
        outOpt << std::setprecision(12);

        for (int iter = 0; iter < rbmAdamIterations; ++iter) {
            RunResult r = runVMC(N, D, optSteps, optEquil, stepParam, mode, baseSeed + 100000 * iter,
                 false, useInteraction, a, b, W, gamma);

            rbmOptimizer.update(a, b, W, r.gradienta, r.gradientb, r.gradientw);
            outOpt << iter << " " << r.energy << " " << r.acceptance << " " << r.cpu_seconds << "\n";

            std::cout << "RBM Adam iter " << iter << " | E = " << r.energy << " | acceptance = " << r.acceptance << "\n";
        }

        outOpt.close();
        std::cout << "Saved RBM Adam optimization path to: " << OptFile << "\n";  
    } else {
        std::cout << "No optimization applied\n ";
    }

    // ---------------- Production run ----------------

    std::string prodFile = txtDir + "energy" + modelTag + modeToString(mode);
    if (mode == Mode::Importance) prodFile += "_dt_" + doubleToTag(dt);
    if (mode == Mode::BruteForce) prodFile += "_sl_" + doubleToTag(stepLength);
    prodFile += "_lr_" + doubleToTag(rbmLearningRate)+ "_Nh_" + std::to_string(Nh);
    if (gamma != 1.0) prodFile += "_gamma_" + doubleToTag(gamma);
    prodFile += "_N" + std::to_string(N) + "_D" + std::to_string(D) + ".txt";

    std::string histFile = txtDir + "e_history" + modelTag + modeToString(mode);
    if (mode == Mode::Importance) histFile += "_dt_" + doubleToTag(dt);
    if (mode == Mode::BruteForce) histFile += "_sl_" + doubleToTag(stepLength);
    histFile += "_lr_" + doubleToTag(rbmLearningRate)+ "_Nh_" + std::to_string(Nh);
    if (gamma != 1.0) histFile
     += "_gamma_" + doubleToTag(gamma);
    histFile += "_N" + std::to_string(N) + "_D" + std::to_string(D) + ".txt";

    std::string gradientFile = txtDir + "gradients" + modelTag + modeToString(mode);
    if (mode == Mode::Importance) gradientFile += "_dt_" + doubleToTag(dt);
    if (mode == Mode::BruteForce) gradientFile += "_sl_" + doubleToTag(stepLength);
    gradientFile += "_lr_" + doubleToTag(rbmLearningRate)+ "_Nh_" + std::to_string(Nh);
    if (gamma != 1.0) gradientFile += "_gamma_" + doubleToTag(gamma);
    gradientFile += "_N" + std::to_string(N) + "_D" + std::to_string(D) + ".txt";

    if (useOptimization) {
        std::cout << "\n=== Production run with the optimal parameters: a, b, W" << " ===\n";
    } else {
        std::cout << "\n=== Production run with default parameters: a, b, W" << " ===\n";
    }

    ParallelRunResult prodPar = runVMCReplicasParallel(N, D, prodSteps, prodEquil, stepParam, mode,
        baseSeed + 9999, nReplicas, true, useInteraction, a, b, W, gamma);

    std::ofstream outProd(prodFile);
    if (!outProd) {
        std::cerr << "Error: could not open " << prodFile << "\n";
        return 1;
    }

    std::ofstream outGrad(gradientFile);
    outGrad << "Gradients\n";
    outGrad << "=========\n\n";
    // Write a
    outGrad << "Gradient_a (" << prodPar.mean.gradienta.size() << " x ";
    if (!prodPar.mean.gradienta.empty()) outGrad << prodPar.mean.gradienta[0].size();
    else outGrad << 0;
    outGrad << ")\n";
    for (size_t i = 0; i < prodPar.mean.gradienta.size(); ++i) {
        outGrad << "Gradient_a[" << i << "] = ";
        for (size_t d = 0; d < prodPar.mean.gradienta[i].size(); ++d) {
            outGrad << prodPar.mean.gradienta[i][d];
            if (d + 1 < prodPar.mean.gradienta[i].size()) outGrad << " ";
        }
        outGrad << "\n";
    }
    outGrad << "\n";
    // Write b
    outGrad << "Gradient_b (" << b.size() << ")\n";
    outGrad << "Gradient_b = ";
    for (size_t j = 0; j < prodPar.mean.gradientb.size(); ++j) {
        outGrad << prodPar.mean.gradientb[j];
        if (j + 1 < prodPar.mean.gradientb.size()) outGrad << " ";
    }
    outGrad << "\n\n";
    // Write W
    outGrad << "Gradient_W (" << prodPar.mean.gradientw.size() << " x ";
    if (!prodPar.mean.gradientw.empty()) outGrad << prodPar.mean.gradientw[0].size();
    else outGrad << 0;
    outGrad << " x ";
    if (!prodPar.mean.gradientw.empty() && !prodPar.mean.gradientw[0].empty()) outGrad << prodPar.mean.gradientw[0][0].size();
    else outGrad << 0;
    outGrad << ")\n";
    for (size_t i = 0; i < prodPar.mean.gradientw.size(); ++i) {
        outGrad << "Gradient_W[" << i << "] =\n";
        for (size_t d = 0; d < prodPar.mean.gradientw[i].size(); ++d) {
            outGrad << "  ";
            for (size_t j = 0; j < prodPar.mean.gradientw[i][d].size(); ++j) {
                outGrad << prodPar.mean.gradientw[i][d][j];
                if (j + 1 < prodPar.mean.gradientw[i][d].size()) outGrad << " ";
            }
            outGrad << "\n";
        }
        outGrad << "\n";
    }
    outGrad.close();

    outProd << "# N D mode stepParam replicas energy acceptance cpu_seconds energy_std energy_stderr\n";
    outProd << N << " " << D << " " << modeToString(mode) << " "
        << stepParam << " "
        << nReplicas << " "
        << std::setprecision(12)
        << prodPar.mean.energy << " "
        << prodPar.mean.acceptance << " "
        << prodPar.mean.cpu_seconds << " "
        << prodPar.energy_std << " "
        << prodPar.energy_stderr << "\n";

    outProd.close();

    std::cout << "\nSaved production result to: " << prodFile << "\n";
    std::cout << "Saved gradient result to: " << gradientFile << "\n";

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
 
/*
    std::string positionsFile = writeProductionParticlePositions(
        N,
        D,
        prodSteps,
        prodEquil,
        stepParam,
        mode,
        baseSeed + 424242,
        useInteraction,
        a,
        b,
        W,
        gamma,
        "data",
        1.0
    );

    std::cout << "Saved particle positions to: " << positionsFile << "\n";
*/

    return 0;
}