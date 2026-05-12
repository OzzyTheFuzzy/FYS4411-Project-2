#pragma once

#include <string>
#include <utility>
#include <vector>

// Forward declarations are enough here
class System;

// Simulation mode
enum class Mode { BruteForce, Importance };

// How the local energy is evaluated in the production run
enum class EvalType { Analytic, Numerical };

// Result returned by one VMC run
struct RunResult {
    double energy = 0.0;
    double gradient = 0.0;   // dE/dalpha from the VMC gradient estimator
    std::vector<std::vector<double>> gradienta;
    std::vector<double> gradientb;
    std::vector<std::vector<std::vector<double>>> gradientw;
    double cpu_seconds = 0.0;
    double acceptance = 0.0;

    //need to acumulate energy values to compute the statistical error properly.
    std::vector<double> energyHistory;

    std::vector<double> densityZ;
    std::vector<double> densityRho;
    std::vector<double> densityZCenters;
    std::vector<double> densityRhoCenters;
};

// Result returned by several independent replicas run in parallel
struct ParallelRunResult {
    RunResult mean;                    // average over replicas
    double wall_seconds = 0.0;         // real elapsed time of the full parallel section
    double energy_std = 0.0;           // std dev across replica mean energies
    double energy_stderr = 0.0;        // energy_std / sqrt(nReplicas)
    int nReplicas = 1;
    std::vector<RunResult> replicas;   // one RunResult per replica
};

// Helper functions used by main.cpp
std::string modeToString(Mode m);
std::string evalToString(EvalType e);
std::string doubleToTag(double x);

RunResult runVMC(
    unsigned int N,
    unsigned int D,
    unsigned int nMetropolis,
    unsigned int nEquil,
    double omega,
    double alpha,
    double stepParam,            // bf: stepLength, is: dt
    Mode mode,
    int seed,
    bool storeEnergyHistory = false,
    bool useInteraction = false,
    bool useRBM = false,
    const std::vector<std::vector<double>>& a = {},
    const std::vector<double>& b = {},
    const std::vector<std::vector<std::vector<double>>>& W = {},
    double beta = 1.0,
    double gamma = 1.0,
    double hardCoreA = 0.0,
    bool densityOnly = false,
    bool storeDensityHist = false,
    bool useNoJastrow = false,
    unsigned int densityBins = 200,
    double densityZMax = 4.0,
    double densityRhoMax = 4.0
);

ParallelRunResult runVMCReplicasParallel(
    unsigned int N,
    unsigned int D,
    unsigned int nMetropolis,
    unsigned int nEquil,
    double omega,
    double alpha,
    double stepParam,
    Mode mode,
    int baseSeed,
    int nReplicas,
    bool storeEnergyHistory = false,
    bool useInteraction = false,
    bool useRBM = false,
    const std::vector<std::vector<double>>& a = {},
    const std::vector<double>& b = {},
    const std::vector<std::vector<std::vector<double>>>& W = {},
    double beta = 1.0,
    double gamma = 1.0,
    double hardCoreA = 0.0
);

std::vector<std::vector<double>> matr_mult(std::vector<std::vector<double>>& a, std::vector<std::vector<double>>& b);