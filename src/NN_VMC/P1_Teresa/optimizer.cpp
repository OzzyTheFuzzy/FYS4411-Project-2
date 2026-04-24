#include "optimizer.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>

OptimizationResult optimizeAlphaBFGS1D(
    const std::function<ObjectiveResult(double)>& objective,
    double alpha0,
    double alphaMin,
    double alphaMax,
    int maxIter,
    double tolerance,
    bool verbose,
    const std::string& logFile
) {
    OptimizationResult out;

    // Optional output file for storing optimization history
    std::ofstream outLog;
    if (!logFile.empty()) {
        outLog.open(logFile);
        if (!outLog) {
            std::cerr << "Warning: could not open optimizer log file " << logFile << "\n";
        } else {
            outLog << "# iter alpha energy gradient H\n";
            outLog << std::setprecision(12);
        }
    }

    // Clamp starting point to allowed range
    double alpha = std::max(alphaMin, std::min(alpha0, alphaMax));

    // Initial VMC evaluation
    ObjectiveResult current = objective(alpha);

    // In 1D, the inverse Hessian approximation is just a scalar
    double H = 1.0;

    for (int iter = 0; iter < maxIter; iter++) {
        if (verbose) {
            std::cout << "[opt] iter=" << iter
                      << " alpha=" << alpha
                      << " E=" << current.energy
                      << " grad=" << current.gradient
                      << " H=" << H << "\n";
        }

        if (outLog) {
            outLog << iter << " " << alpha << " " 
                   << current.energy << " " << current.gradient << " "
                   << H << "\n";
        }

        // Convergence check
        if (std::abs(current.gradient) < tolerance) {
            out.alpha_opt = alpha;
            out.energy_opt = current.energy;
            out.gradient_opt = current.gradient;
            out.iterations = iter + 1;
            out.converged = true;
            return out;
        }

        // Quasi-Newton search direction
        double p = -H * current.gradient;

        // Simple backtracking line search
        double stepScale = 1.0;
        double alphaNew = alpha;
        ObjectiveResult trial = current;

        bool accepted = false;
        for (int ls = 0; ls < 20; ls++) {
            alphaNew = alpha + stepScale * p;

            // Keep alpha in a reasonable physical interval
            alphaNew = std::max(alphaMin, std::min(alphaNew, alphaMax));

            // If clamping produced no actual move, reduce step
            if (std::abs(alphaNew - alpha) < 1e-12) {
                stepScale *= 0.5;
                continue;
            }

            trial = objective(alphaNew);

            // Accept if energy improved
            if (trial.energy < current.energy) {
                accepted = true;
                break;
            }

            stepScale *= 0.5;
        }

        if (!accepted) {
            // Fall back to a small gradient-descent move
            alphaNew = alpha - 0.005 * current.gradient;
            alphaNew = std::max(alphaMin, std::min(alphaNew, alphaMax));
            trial = objective(alphaNew);
        }

        double s = alphaNew - alpha;
        double y = trial.gradient - current.gradient;

        // 1D BFGS / secant-style inverse Hessian update
        if (std::abs(y) > 1e-12 && s * y > 0.0) {
            H = s / y;
        } else {
            // Reset if curvature information is poor/noisy
            H = 1.0;
        }

        alpha = alphaNew;
        current = trial;

        out.alpha_opt = alpha;
        out.energy_opt = current.energy;
        out.gradient_opt = current.gradient;
        out.iterations = iter + 1;
        out.converged = false;
    }

    return out;
}