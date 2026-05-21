#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <memory>
#include <cassert>
#include <cstring>
#include <armadillo>
#include <nlopt.hpp>

#include "system.h"
#include "common.h"
#include "config.h"
#include "WaveFunctions/simplegaussian.h"
#include "WaveFunctions/ellipticgaussian.h"
#include "WaveFunctions/repulsiveellipticgaussian.h"
#include "WaveFunctions/nn_envelope.h"
#include "Hamiltonians/harmonicoscillator.h"
#include "Hamiltonians/repulsiveho.h"
#include "Hamiltonians/coulombho.h"
#include "InitialStates/initialstate.h"
#include "Solvers/metropolis.h"
#include "Solvers/metropolishastings.h"
#include "Math/random.h"
#include "Math/blocker.h"
#include "particle.h"
#include "onebodydensity.h"
#include "Samplers/energysampler.h"
#include "Samplers/densitysampler.h"
#include "VMCOptimizer.h"
#include "VMCOptimizer_NN.h"

using namespace std;
using namespace CommonUtils;

using HamiltonianFactory = function<unique_ptr<class Hamiltonian>()>;
using WaveFunctionFactory = function<unique_ptr<class WaveFunction>(const vector<double>&)>;
using SolverFactory = function<unique_ptr<MonteCarlo>(unique_ptr<Random>)>;
using ActivationFunc = std::function<torch::Tensor(const torch::Tensor&)>;
using ActivationFuncFactory = std::function<ActivationFunc()>;

void printLogHeader(const runConfig& cfg, std::ofstream& globalLog, tm* now_tm) {
    globalLog << "=========================================\n";
        globalLog << "               VMC RUN LOG               \n";
        globalLog << "=========================================\n";
        globalLog << "Date and time    : " << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S") << "\n";
        globalLog << "-----------------------------------------\n";
        globalLog << "[ SYSTEM & MODELS ]\n";
        globalLog << "Hamiltonian      : " << cfg.hamiltonianType << "\n";
        globalLog << "WF Train Type    : " << cfg.waveFunctionTrainType << "\n";
        globalLog << "WaveFunction     : " << cfg.waveFunctionType << "\n";
        globalLog << "Solver           : " << cfg.solverType << "\n";
        globalLog << "preferAnalytic   : " << (cfg.preferAnalytic ? "true" : "false") << "\n";
        globalLog << "useCache         : " << (cfg.useCache ? "true" : "false") << "\n";
        globalLog << "-----------------------------------------\n";
        globalLog << "[ PHYSICAL PARAMETERS ]\n";
        globalLog << "dimensions (D)   : " << cfg.numberOfDimensions << "\n";
        globalLog << "particles (N)    : " << cfg.numberOfParticles << "\n";
        globalLog << "omega            : " << cfg.omega << "\n";
        globalLog << "omega_z          : " << cfg.omega_z << "\n";
        globalLog << "rep_a_factor     : " << cfg.repulsive_a_factor << "\n";
        globalLog << "rep_strength     : " << cfg.repulsive_strength << "\n";
        globalLog << "maxStrength      : " << cfg.maxStrength << "\n";
        globalLog << "-----------------------------------------\n";
        globalLog << "[ MONTE CARLO & OPTIMIZATION ]\n";
        globalLog << "time Step        : " << cfg.timeStep << "\n";
        globalLog << "Equilibr. Steps  : " << cfg.equilibrationSteps << "\n";
        globalLog << "BFGS MC Steps    : " << cfg.metropolisSteps << "\n";
        globalLog << "BFGS_tol         : " << cfg.BFGS_tol << "\n";
        globalLog << "Final MC Steps   : 2^" << cfg.finalMClog2steps << " (" << std::pow(2, cfg.finalMClog2steps) << ")\n";
        globalLog << "-----------------------------------------\n";
        globalLog << "[ NEURAL NETWORK ]\n";
        globalLog << "Nhid             : " << cfg.Nhid << "\n";
        globalLog << "Activation Func  : " << cfg.activationFunctionType << "\n";
        globalLog << "Learning Rate    : " << cfg.lr << "\n";
        globalLog << "Pretrain Steps   : " << cfg.nPretrainSteps << "\n";
        globalLog << "Energy Steps     : " << cfg.nEnergySteps << "\n";
        globalLog << "Adiab Steps      : " << cfg.nAdiabSteps << "\n";
        globalLog << "Adam_ktol        : " << cfg.Adam_ktol << "\n";
        globalLog << "max_patience     : " << cfg.max_patience << "\n";
        globalLog << "min_improvement  : " << cfg.min_improvement << "\n";
        globalLog << "helpDecay        : " << cfg.helpDecay << "\n";
        globalLog << "-----------------------------------------\n";
        globalLog << "[ OBSERVABLES & MISC ]\n";
        globalLog << "1bodyDens. Steps : " << cfg.onebodyDensitySteps << "\n";
        globalLog << "1bodyDens. rMax  : " << cfg.onebodyDensity_rMax << "\n";
        globalLog << "1bodyDens. nBins : " << cfg.onebodyDensity_nBins << "\n";
        globalLog << "Seed             : " << cfg.seed << "\n";
        globalLog << "=========================================\n\n";
        globalLog << std::flush;
}

int main(int argc, char* argv[]) {
    runConfig cfg = loadConfig("config.json");

    /*
    // --- Parameters ---
    string hamiltonianType = "CoulombHO";  // HarmonicOscillator or RepulsiveHO or CoulombHO
    string waveFunctionTrainType = "EllipticGaussian"; // SimpleGaussian or EllipticGaussian or RepEllipticGaussian
    string waveFunctionType = "NN_envelope"; // SimpleGaussian or EllipticGaussian or RepEllipticGaussian or NN_envelope
    string solverType = "MetropolisHastings"; // Metropolis or MetropolisHastings
    bool preferAnalytic = false;
    bool useCache = false;
    unsigned int numberOfDimensions = 3;
    unsigned int numberOfParticles = 3;
    unsigned int numberOfMetropolisSteps = 10e3;
    unsigned int numberOfEquilibrationSteps = 1e4;
    unsigned int finalMClog2steps = log2(1e5);
    unsigned int onebodyDensitySteps = 1e5;
    double omega = 1.0;
    double omega_z = 2.8243;
    double repulsive_a_factor = 0.0043;
    double repulsive_strength = numeric_limits<double>::infinity();
    double timeStep = 0.15;     // for brute force Metropolis, this corresponds to stepLength
    double onebodyDensity_rMax = 3.5;
    unsigned int onebodyDensity_nBins = 50;
    double BFGS_tol = 1e-5;     // NLopt's xtol_rel relative tolerance criterion for optimization
    // Next ones are parameters relevant for NNs
    int Nhid = 12;
    string activationFunctionType = "tanh"; // gelu or tanh or relu or sigmoid
    double helpDecay = 0.4; // typically 0 < helpDecay <= 0.5
    double lr = 5e-2;     // 5e-2 looked good
    double Adam_ktol = 0.99;
    int max_patience = 80;
    double min_improvement = 0.05;
    int nPretrainSteps = 1000;   // maximize K
    int nEnergySteps = 7000;  // minimize E
    int nAdiabSteps = 1000;  // adiabatic interaction addition
    double maxStrength = 1;   // maximum coulomb potential strength

    // int seed = 0;    // if seed == 0, seed is chosen randomly at each RNG construction
    int seed = chrono::system_clock::now().time_since_epoch().count();
    // vector<double> initialParams = { 0.75 };
    vector<double> initialParams = { 0.50, 2.8243 };
    */

    chrono::high_resolution_clock::time_point watch_start, watch_end;
    chrono::duration<double> elapsedTime;

    // --- Toggles based on argc, argv ---
    vector<bool> toggles(3, false);
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            int temp = atoi(argv[i]);
            if (0 < temp && temp <= (int)toggles.size()) {
                toggles[temp - 1] = true;
            }
        }
    }
    else {
        toggles.assign(toggles.size(), true);
    }

    // --- Global Log file setup ---
    auto now = chrono::system_clock::now();
    time_t now_time = chrono::system_clock::to_time_t(now);
    tm* now_tm = localtime(&now_time);
    ostringstream filenameStream;
    filenameStream << "./logs/run_" << put_time(now_tm, "%Y%m%d_%H%M%S") << ".log";
    ofstream globalLog(filenameStream.str());
    if (!globalLog.is_open()) {
        cerr << "Error: unable to generate log file " << filenameStream.str() << endl;
        return 1;
    }
    printLogHeader(cfg, globalLog, now_tm);

    // --- Setup Factories ---
    HamiltonianFactory hFac = [=]() -> unique_ptr<Hamiltonian> {
        if (cfg.hamiltonianType == "HarmonicOscillator") {
            return make_unique<HarmonicOscillator>(cfg.omega);
        }
        else if (cfg.hamiltonianType == "CoulombHO") {
            return make_unique<CoulombHO>(cfg.omega, cfg.omega_z, cfg.maxStrength);
        }
        else { // default to Repulsive
            return make_unique<RepulsiveHO>(cfg.omega, cfg.omega_z, cfg.repulsive_a_factor, cfg.repulsive_strength);
        }
        };
    WaveFunctionFactory wfFac = [=](const vector<double>& p) -> unique_ptr<WaveFunction> {
        if (cfg.waveFunctionType == "SimpleGaussian")
            return make_unique<SimpleGaussian>(p[0]);
        else if (cfg.waveFunctionType == "EllipticGaussian")
            return make_unique<EllipticGaussian>(p[0], p[1]);
        else if (cfg.waveFunctionType == "NN_envelope")
            return make_unique<NN_envelope>(cfg.numberOfParticles,
                cfg.numberOfDimensions,
                cfg.numberOfParticles * cfg.numberOfDimensions,
                cfg.Nhid,
                cfg.helpDecay,
                p
            );
        else // default to Repulsive
            return make_unique<RepEllipticGaussian>(p[0], p[1], cfg.repulsive_a_factor / sqrt(cfg.omega));
        };
    WaveFunctionFactory wfFacTrain = [=](const vector<double>& p) -> unique_ptr<WaveFunction> {
        if (cfg.waveFunctionTrainType == "SimpleGaussian")
            return make_unique<SimpleGaussian>(p[0]);
        else if (cfg.waveFunctionTrainType == "EllipticGaussian")
            return make_unique<EllipticGaussian>(p[0], p[1]);
        else // default to Repulsive
            return make_unique<RepEllipticGaussian>(p[0], p[1], cfg.repulsive_a_factor / sqrt(cfg.omega));
        };
    SolverFactory solverFac = [=](unique_ptr<Random> rng) -> unique_ptr<MonteCarlo> {
        if (cfg.solverType == "Metropolis") {
            return make_unique<Metropolis>(move(rng), cfg.preferAnalytic, cfg.useCache);
        }
        else // default to Metropolis-Hastings
            return make_unique<MetropolisHastings>(move(rng), cfg.preferAnalytic, cfg.useCache);
        };
    ActivationFuncFactory actFun = [=]() -> ActivationFunc {
        if (cfg.activationFunctionType == "gelu") {
            return [](const torch::Tensor& t) { return torch::gelu(t); };
        }
        else if (cfg.activationFunctionType == "relu") {
            return [](const torch::Tensor& t) { return torch::relu(t); };
        }
        else if (cfg.activationFunctionType == "sigmoid") {
            return [](const torch::Tensor& t) { return torch::sigmoid(t); };
        }
        else {
            // default to tanh
            return [](const torch::Tensor& t) { return torch::tanh(t); };
        }
        };

    if (toggles[0] && cfg.waveFunctionType == "NN_envelope") {
        // --- 1a: Neural-Network optimization ---
        globalLog << "Initial parameters for psi_train: " << setprecision(9);
        for (unsigned int i = 0; i < cfg.initialParams.size(); i++) {
            globalLog << cfg.initialParams[i] << ", \t";
        }
        globalLog << endl;

        ostringstream filenameStream;
        filenameStream << "./logs_NN/run_" << put_time(now_tm, "%Y%m%d_%H%M%S") << ".csv";
        ofstream logfile(filenameStream.str());
        if (!logfile.is_open()) {
            cerr << "Error: unable to generate log_NN file " << filenameStream.str() << endl;
            return 1;
        }
        ofstream outfile("./iofiles/details_results.csv");
        ofstream paramsfile("./iofiles/params.dat");
        VMCOptimizer_NN optimizer(
            cfg,
            hFac(),
            solverFac,
            actFun(),
            & logfile, & outfile, & paramsfile
        );

        watch_start = chrono::high_resolution_clock::now();
        vector<double> optimalParams = optimizer.optimize(wfFacTrain(cfg.initialParams));
        watch_end = chrono::high_resolution_clock::now();
        elapsedTime = watch_end - watch_start;

        cout << "Optimal parameters: " << setprecision(9);
        globalLog << "Optimal parameters: " << setprecision(9);
        for (unsigned int i = 0; i < optimalParams.size(); i++) {
            cout << optimalParams[i] << ", \t";
            globalLog << optimalParams[i] << ", \t";
        }
        cout << "\nNN_VMC optimization done (in " << elapsedTime.count() << " s).\n\n";
        globalLog << "\nNN_VMC Optimization done (in " << elapsedTime.count() << " s).\n\n";
        logfile.close(); outfile.close(); paramsfile.close();
    }

    // --- Build MC engine ---
    MCEngine engine(
        cfg.numberOfDimensions,
        cfg.numberOfParticles,
        cfg.equilibrationSteps,
        cfg.timeStep,
        hFac,
        wfFac,
        solverFac,
        cfg.seed
    );

    if (toggles[0] && cfg.waveFunctionType != "NN_envelope") {
        // --- 1b: Optimization ---
        globalLog << "Initial parameters: " << setprecision(9);
        for (unsigned int i = 0; i < cfg.initialParams.size(); i++) {
            globalLog << cfg.initialParams[i] << ", \t";
        }
        globalLog << endl;

        ofstream logfile("./iofiles/log.csv");
        ofstream outfile("./iofiles/details_results.csv");
        ofstream paramsfile("./iofiles/params.dat");
        VMCOptimizer optimizer(engine, cfg.metropolisSteps, cfg.BFGS_tol, &logfile, &outfile, &paramsfile);

        watch_start = chrono::high_resolution_clock::now();
        vector<double> optimalParams = optimizer.optimize(cfg.initialParams);
        watch_end = chrono::high_resolution_clock::now();
        elapsedTime = watch_end - watch_start;

        cout << "Optimal parameters: " << setprecision(9);
        globalLog << "Optimal parameters: " << setprecision(9);
        for (unsigned int i = 0; i < optimalParams.size(); i++) {
            cout << optimalParams[i] << ", \t";
            globalLog << optimalParams[i] << ", \t";
        }
        cout << "\nVMC Optimization done (in " << elapsedTime.count() << " s).\n\n";
        globalLog << "\nVMC Optimization done (in " << elapsedTime.count() << " s).\n\n";
        logfile.close(); outfile.close(); paramsfile.close();
    }

    if (toggles[1]) {
        // --- 2a: Final MC ---
        watch_start = chrono::high_resolution_clock::now();
        vector<double> params = readVector("./iofiles/params.dat");
        ofstream energiesfile("./iofiles/finalMCenergies.dat");
        unique_ptr<EnergySampler> sampler =
            engine.run(params, (unsigned int)pow(2, cfg.finalMClog2steps), &energiesfile);
        cout << scientific << setprecision(9) << "FinalMC energy: " << sampler->getEnergy()
            << " +- " << sampler->getError() << endl << defaultfloat;
        globalLog << scientific << setprecision(9) << "FinalMC energy: " << sampler->getEnergy()
            << " +- " << sampler->getError() << endl << defaultfloat;
        cout << scientific << setprecision(9) << "Acceptance ratio: " << sampler->getAcceptanceRatio()
            << endl << defaultfloat;
        globalLog << scientific << setprecision(9) << "Acceptance ratio: " << sampler->getAcceptanceRatio()
            << endl << defaultfloat;
        watch_end = chrono::high_resolution_clock::now();
        elapsedTime = watch_end - watch_start;
        cout << "FinalMC done (in " << elapsedTime.count() << " s).\n\n";
        globalLog << "FinalMC done (in " << elapsedTime.count() << " s).\n\n";
        energiesfile.close();

        // --- 2b: Blocking ---
        watch_start = chrono::high_resolution_clock::now();
        vector<double> finalEnergies = readVector("./iofiles/finalMCenergies.dat");
        Blocker block(finalEnergies);
        block.printResults("./iofiles/blocking_results.csv");
        cout << scientific << setprecision(9) << "Blocking energy: " << block.mean
            << " +- " << block.stdErr << endl << defaultfloat;
        globalLog << scientific << setprecision(9) << "Blocking energy: " << block.mean
            << " +- " << block.stdErr << endl << defaultfloat;
        watch_end = chrono::high_resolution_clock::now();
        elapsedTime = watch_end - watch_start;
        cout << "Blocking analysis done (in " << elapsedTime.count() << " s).\n\n";
        globalLog << "Blocking analysis done (in " << elapsedTime.count() << " s).\n\n";
    }

    if (toggles[2]) {
        // --- 3: One-body density ---
        watch_start = chrono::high_resolution_clock::now();
        vector<double> params = readVector("./iofiles/params.dat");
        vector<vector<double>> rGrid = readMatrix("./iofiles/rGrid.csv");
        ostringstream filenameStream;
        filenameStream << "./logs_OBD/run_" << put_time(now_tm, "%Y%m%d_%H%M%S") << ".csv";
        ofstream densityfile(filenameStream.str());
        vector<pair<double, double>> density = computeOnebodyDensity(
            engine, params, cfg.onebodyDensitySteps, cfg.onebodyDensity_rMax,
            cfg.onebodyDensity_nBins, &densityfile);
        watch_end = chrono::high_resolution_clock::now();
        elapsedTime = watch_end - watch_start;
        cout << "One-body density done (in " << elapsedTime.count() << " s).\n\n";
        globalLog << "One-body density done (in " << elapsedTime.count() << " s).\n\n";
        densityfile.close();
    }


    globalLog << "=========================================\n";
    globalLog.close();

    cout << "Exiting. (Log saved in: " << filenameStream.str() << ")\n";
    return 0;
}