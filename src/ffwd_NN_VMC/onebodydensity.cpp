#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "onebodydensity.h"
#include "common.h"
#include "Math/random.h"
#include "Samplers/densitysampler.h"

using namespace CommonUtils;

std::vector<std::pair<double, double>> computeOnebodyDensity(
    MCEngine& engine,
    const std::vector<double>& params,
    unsigned int numberOfMetropolisSteps,
    double rMax,
    unsigned int nBins,
    std::ofstream* densitiesOut) {

    std::cout << "Computing one-body density..." << std::flush;

    std::unique_ptr<DensitySampler> sampler = engine.runOnebodyDensity(params, numberOfMetropolisSteps, rMax, nBins);

    const std::vector<double>& grid = sampler->getRadialGrid();
    const std::vector<double>& densityProf = sampler->getDensityProfile();
    const std::vector<double>& probabilityProf = sampler->getProbabilityProfile();
    const std::vector<double>& densityErrProf = sampler->getDensityError();
    const std::vector<double>& probabilityErrProf = sampler->getProbabilityError();

    std::vector<std::pair<double, double>> result(nBins);

    if (densitiesOut) {
        *densitiesOut << "# r, density, density error estimate, radial probability, probability error estimate\n";
    }

    for (unsigned int i = 0; i < nBins; i++) {
        result[i] = std::make_pair(grid[i], densityProf[i]);
        if (densitiesOut) {
            *densitiesOut << std::scientific << std::setprecision(9) << grid[i]
                << ", " << densityProf[i] << ", "
                << densityErrProf[i] << ", " << probabilityProf[i] << ", "
                << probabilityErrProf[i] << std::endl;
        }
    }

    std::cout << "\r                             " << std::endl;

    return result;
}