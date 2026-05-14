#pragma once

#include <vector>
#include <fstream>
#include "WaveFunctions/wavefunction.h"
#include "particle.h"
#include "mcengine.h"
#include "Math/random.h"

/**
 * @brief Computes the one-body density of the quantum system.
 *
 * This function triggers a dedicated Variational Monte Carlo simulation 
 * focused exclusively on sampling the radial density using a histogram approach.
 * Particle positions are averaged over hyperspherical shells.
 *
 * @param engine The pre-configured MCEngine instance.
 * @param params Vector containing optimal variational parameters.
 * @param numberOfMetropolisSteps Total number of simulation steps.
 * @param rMax Maximum spatial radius (in units of \f$a_{ho}\f$) for the histogram.
 * @param nBins Total number of bins to divide the \f$[0, rMax]\f$ interval into.
 * @param densitiesOut Optional pointer to an ofstream to save the raw histogram data.
 * @return std::vector<std::pair<double, double>> A vector of pairs containing:
 * - first: radial distance r (center of the bin).
 * - second: local density \f$\rho(r)\f$ evaluated at that point.
 */
std::vector<std::pair<double, double>> computeOnebodyDensity(
    MCEngine& engine,
    const std::vector<double>& params,
    unsigned int numberOfMetropolisSteps,
    double rMax,
    unsigned int nBins,
    std::ofstream* densitiesOut = nullptr);