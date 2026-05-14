#pragma once
#include <vector>
#include <fstream>
#include <string>

#include "mcengine.h"

/**
 * @brief Class responsible for the automatic optimization of variational parameters.
 * * Leverages the NLopt library to perform an optimization (e.g., BFGS or Nelder-Mead) 
 * to minimize the ground state energy.
 */
class VMCOptimizer {
public:
    /**
     * @brief Constructs the optimizer wrapper.
     * @param engine Reference to the main MCEngine.
     * @param numberOfMetropolisSteps Steps per optimization iteration.
     * @param BFGS_tol Tolerance for stopping the optimization algorithm.
     * @param logfile Optional pointer to log optimization progress.
     * @param outfile Optional pointer to output general results.
     * @param paramsfile Optional pointer to output final parameters to a dedicated file.
     */
    VMCOptimizer(
        MCEngine& engine,
        unsigned int numberOfMetropolisSteps,
        double BFGS_tol,
        std::ofstream* logfile = nullptr,
        std::ofstream* outfile = nullptr,
        std::ofstream* paramsfile = nullptr
    );

    /**
     * @brief Runs the optimization starting from an initial guess.
     * @param initialParams Vector containing the starting variational parameters.
     * @return Vector containing the optimal parameters found.
     */
    std::vector<double> optimize(std::vector<double> initialParams);

private:
    /**
     * @brief Runs a single VMC pass to compute energy and fill the gradient array.
     * @param params Current variational parameters.
     * @param grad Vector to be filled with the energy gradient (if algorithm requires it).
     * @return Computed local energy for the given parameters.
     */
    double computeMC(const std::vector<double>& params, std::vector<double>& grad);

    /**
     * @brief Static wrapper required by the NLopt C-style callback function.
     * @param params Array of parameters provided by NLopt.
     * @param grad Array to write the gradient to.
     * @param data Void pointer to the instance of VMCOptimizer.
     * @return The objective function value (energy).
     */
    static double nloptObjective(const std::vector<double>& params,
        std::vector<double>& grad,
        void* data) {
        return static_cast<VMCOptimizer*>(data)->computeMC(params, grad);
    }

    MCEngine& m_engine;
    unsigned int m_numberOfMetropolisSteps;
    double m_BFGS_tol;
    std::ofstream* m_logfile;
    std::ofstream* m_outfile;
    std::ofstream* m_paramsfile;
    unsigned int m_mcCount = 0;
};