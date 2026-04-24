#pragma once
#include <functional>
#include <string>

// Result returned by one VMC evaluation inside the optimizer
struct ObjectiveResult {
    double energy = 0.0;
    double gradient = 0.0;
};

// Final result returned by the optimizer
struct OptimizationResult {
    double alpha_opt = 0.5;
    double energy_opt = 0.0;
    double gradient_opt = 0.0;
    int iterations = 0;
    bool converged = false;
};

// One-dimensional quasi-Newton / BFGS-style optimizer.
// The callback must return both E(alpha) and dE/dalpha.
OptimizationResult optimizeAlphaBFGS1D(
    const std::function<ObjectiveResult(double)>& objective,
    double alpha0,
    double alphaMin,
    double alphaMax,
    int maxIter,
    double tolerance,
    bool verbose = true,
    const std::string& logFile = ""
);