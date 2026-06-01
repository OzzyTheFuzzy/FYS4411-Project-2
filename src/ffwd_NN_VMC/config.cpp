#include <fstream>
#include <iostream>
#include "config.h"
#include "json.hpp" // The single-header library

using json = nlohmann::json;

// A function that returns a fully populated Config struct
#include "config.h"
#include "json.hpp" // The nlohmann/json header
#include <fstream>
#include <iostream>
#include <limits>

using json = nlohmann::json;

runConfig loadConfig(const std::string& filepath) {
    runConfig cfg;
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << filepath << "!\n";
        std::cerr << "Using default hardcoded parameters.\n";
        return cfg; 
    }

    json j;
    try {
        file >> j;
    } catch (const json::parse_error& e) {
        std::cerr << "JSON Parsing Error: " << e.what() << "\n";
        return cfg;
    }

    // --- [ SYSTEM & MODELS ] ---
    if (j.contains("system")) {
        cfg.hamiltonianType       = j["system"].value("hamiltonianType", cfg.hamiltonianType);
        cfg.waveFunctionTrainType = j["system"].value("waveFunctionTrainType", cfg.waveFunctionTrainType);
        cfg.waveFunctionType      = j["system"].value("waveFunctionType", cfg.waveFunctionType);
        cfg.solverType            = j["system"].value("solverType", cfg.solverType);
        cfg.preferAnalytic        = j["system"].value("preferAnalytic", cfg.preferAnalytic);
        cfg.useCache              = j["system"].value("useCache", cfg.useCache);
    }

    // --- [ PHYSICAL PARAMETERS ] ---
    if (j.contains("physics")) {
        cfg.numberOfDimensions = j["physics"].value("dimensions", cfg.numberOfDimensions);
        cfg.numberOfParticles  = j["physics"].value("particles", cfg.numberOfParticles);
        cfg.omega              = j["physics"].value("omega", cfg.omega);
        cfg.omega_z            = j["physics"].value("omega_z", cfg.omega_z);
        cfg.repulsive_a_factor = j["physics"].value("repulsive_a_factor", cfg.repulsive_a_factor);
        cfg.maxStrength        = j["physics"].value("maxStrength", cfg.maxStrength);

        // Special handling for the "inf" string
        if (j["physics"].contains("repulsive_strength")) {
            if (j["physics"]["repulsive_strength"] == "inf") {
                cfg.repulsive_strength = std::numeric_limits<double>::infinity();
            } else {
                cfg.repulsive_strength = j["physics"]["repulsive_strength"];
            }
        }

        // Parse the initial parameters vector directly
        if (j["physics"].contains("initialParams")) {
            cfg.initialParams = j["physics"]["initialParams"].get<std::vector<double>>();
        }
    }

    // --- [ MONTE CARLO ] ---
    if (j.contains("monte_carlo")) {
        cfg.timeStep           = j["monte_carlo"].value("timeStep", cfg.timeStep);
        cfg.equilibrationSteps = j["monte_carlo"].value("equilibrationSteps", cfg.equilibrationSteps);
        cfg.metropolisSteps    = j["monte_carlo"].value("metropolisSteps", cfg.metropolisSteps);
        cfg.finalMClog2steps   = j["monte_carlo"].value("finalMClog2steps", cfg.finalMClog2steps);
        cfg.BFGS_tol           = j["monte_carlo"].value("BFGS_tol", cfg.BFGS_tol);
    }

    // --- [ NEURAL NETWORK ] ---
    if (j.contains("neural_network")) {
        cfg.Nhid                   = j["neural_network"].value("Nhid", cfg.Nhid);
        cfg.activationFunctionType = j["neural_network"].value("activationFunctionType", cfg.activationFunctionType);
        cfg.helpDecay              = j["neural_network"].value("helpDecay", cfg.helpDecay);
        cfg.lr                     = j["neural_network"].value("lr", cfg.lr);
        cfg.Adam_ktol              = j["neural_network"].value("Adam_ktol", cfg.Adam_ktol);
        cfg.max_patience           = j["neural_network"].value("max_patience", cfg.max_patience);
        cfg.min_improvement        = j["neural_network"].value("min_improvement", cfg.min_improvement);
        cfg.nPretrainSteps         = j["neural_network"].value("nPretrainSteps", cfg.nPretrainSteps);
        cfg.nEnergySteps           = j["neural_network"].value("nEnergySteps", cfg.nEnergySteps);
        cfg.nAdiabSteps            = j["neural_network"].value("nAdiabSteps", cfg.nAdiabSteps);
    }

    // --- [ OBSERVABLES ] ---
    if (j.contains("observables")) {
        cfg.onebodyDensitySteps  = j["observables"].value("onebodyDensitySteps", cfg.onebodyDensitySteps);
        cfg.onebodyDensity_rMax  = j["observables"].value("onebodyDensity_rMax", cfg.onebodyDensity_rMax);
        cfg.onebodyDensity_nBins = j["observables"].value("onebodyDensity_nBins", cfg.onebodyDensity_nBins);
    }

    // --- [ MISC ] ---
    if (j.contains("misc")) {
        cfg.seed = j["misc"].value("seed", cfg.seed);
    }

    return cfg;
}