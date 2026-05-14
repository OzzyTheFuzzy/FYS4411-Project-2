#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>
#include "system.h"
#include "common.h"
#include "sampler.h"
#include "particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"

using namespace CommonUtils;

Sampler::Sampler(
    unsigned int numberOfParticles,
    unsigned int numberOfDimensions,
    unsigned int numberOfParameters,
    double stepLength,
    unsigned int numberOfMetropolisSteps
) {
    m_stepNumber = 0;
    m_numberOfMetropolisSteps = numberOfMetropolisSteps;
    m_numberOfParticles = numberOfParticles;
    m_numberOfDimensions = numberOfDimensions;
    m_numberOfParameters = numberOfParameters;
    m_stepLength = stepLength;
    m_numberOfAcceptedSteps = 0;
    m_watch_start = std::chrono::high_resolution_clock::now();
}