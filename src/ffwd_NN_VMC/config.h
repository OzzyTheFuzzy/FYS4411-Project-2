// config.h
#pragma once
#include <string>
#include <limits>
#include <vector>

using namespace std;

struct runConfig {
    // [ SYSTEM & MODELS ]
    string hamiltonianType = "CoulombHO";
    string waveFunctionTrainType = "EllipticGaussian";
    string waveFunctionType = "NN_envelope";
    string solverType = "MetropolisHastings";
    bool preferAnalytic = false;
    bool useCache = false;

    // [ PHYSICAL PARAMETERS ]
    unsigned int numberOfDimensions = 2;
    unsigned int numberOfParticles = 2;
    double omega = 1.0;
    double omega_z = 2.8243;
    double repulsive_a_factor = 0.0043;
    double repulsive_strength = numeric_limits<double>::infinity();
    double maxStrength = 1.0;
    vector<double> initialParams = { 0.50, 2.8243 };

    // [ MONTE CARLO ]
    double timeStep = 0.15;
    unsigned int equilibrationSteps = 10000;
    unsigned int metropolisSteps = 10000;
    unsigned int finalMClog2steps = 16;
    double BFGS_tol = 1e-5;

    // [ NEURAL NETWORK ]
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
    
    // [ ONEBODY DENSITY ]
    unsigned int onebodyDensitySteps = 1e5;
    double onebodyDensity_rMax = 3.5;
    unsigned int onebodyDensity_nBins = 50;

    // [ MISC ]
    int seed = 0; // Set later
};

runConfig loadConfig(const std::string& filepath);