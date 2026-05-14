#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>
#include "system.h"
#include "common.h"
#include "energysampler.h"
#include "particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"

using namespace CommonUtils;

EnergySampler::EnergySampler(
    unsigned int numberOfParticles,
    unsigned int numberOfDimensions,
    unsigned int numberOfParameters,
    double stepLength,
    unsigned int numberOfMetropolisSteps
) : Sampler(numberOfParticles,
    numberOfDimensions,
    numberOfParameters,
    stepLength,
    numberOfMetropolisSteps) {
    m_covariance.resize(m_numberOfParameters, 0);
    m_opO.resize(m_numberOfParameters, 0);
    m_energy = 0;
    m_energySQ = 0;
    m_variance = 0;
    m_error = 0;
    m_cumulativeEnergy = 0;
    m_cumulativeEnergySQ = 0;
}


void EnergySampler::sample(bool acceptedStep, System* system, std::ofstream* energiesOut) {
    /* Here you should sample all the interesting things you want to measure.
     * Note that there are (way) more than the single one here currently.
     */
    double localEnergy = system->computeLocalEnergy();
    if (energiesOut != nullptr) {
        *energiesOut << localEnergy << '\n';
    }
    m_cumulativeEnergy += localEnergy;
    m_cumulativeEnergySQ += sq(localEnergy);
    std::vector<double> OW = system->computeLogParDer_vect();
    for (unsigned int i = 0; i < m_numberOfParameters; i++) {
        m_opO[i] += OW[i];
        m_covariance[i] += OW[i] * localEnergy;
    }
    m_stepNumber++;
    m_numberOfAcceptedSteps += acceptedStep;
    m_watch_end = std::chrono::high_resolution_clock::now();
}

void EnergySampler::printOutputToTerminal(System& system) {
    std::cout << std::endl;
    std::cout << "  -- System info -- " << std::endl;
    std::cout << " Number of particles  : " << m_numberOfParticles << std::endl;
    std::cout << " Number of dimensions : " << m_numberOfDimensions << std::endl;
    std::cout << " Number of Metropolis steps run : 10^" << std::log10(m_numberOfMetropolisSteps) << std::endl;
    std::cout << " Step length used : " << m_stepLength << std::endl;
    std::cout << " Ratio of accepted steps: " << ((double)m_numberOfAcceptedSteps) / ((double)m_numberOfMetropolisSteps) << std::endl;
    std::cout << " Elapsed time: " << m_elapsedTime.count() << " s\n";
    std::cout << std::endl;
    std::cout << "  -- Wave function parameters -- " << std::endl;
    std::cout << " Number of parameters : " << m_numberOfParameters << std::endl;
    for (unsigned int i = 0; i < m_numberOfParameters; i++) {
        std::cout << " Parameter " << i + 1 << " : " << system.getWaveFunctionParameters()[i] << std::endl;
    }
    std::cout << std::endl;
    std::cout << "  -- Results -- " << std::endl;
    std::cout << " Energy : " << m_energy << std::endl;
    std::cout << " Variance : " << m_variance << std::endl;
    std::cout << " Error : " << m_error << std::endl;
    std::cout << std::endl;
}

void EnergySampler::printOutputToFile(System& system, std::ofstream& outs) {
    outs << std::endl;
    outs << "#  -- System info -- " << std::endl;
    outs << "# Number of particles  : " << m_numberOfParticles << std::endl;
    outs << "# Number of dimensions : " << m_numberOfDimensions << std::endl;
    outs << "# Number of Metropolis steps run : 10^" << std::log10(m_numberOfMetropolisSteps) << std::endl;
    outs << "# Step length used : " << m_stepLength << std::endl;
    outs << "# Ratio of accepted steps: " << ((double)m_numberOfAcceptedSteps) / ((double)m_numberOfMetropolisSteps) << std::endl;
    outs << "# Elapsed time: " << m_elapsedTime.count() << " s\n";
    outs << std::endl;
    outs << "#  -- Wave function parameters -- " << std::endl;
    outs << "# Number of parameters : " << m_numberOfParameters << "\n#";
    for (unsigned int i = 0; i < m_numberOfParameters; i++) {
        outs << " p[" << i << "],  \t ";
    }
    outs << " energy,  \t  variance,  \t  error\n";
    for (unsigned int i = 0; i < m_numberOfParameters; i++) {
        outs << system.getWaveFunctionParameters()[i] << ", \t";
    }
    outs << std::setprecision(10);
    outs << m_energy << ", \t" << m_variance << ", \t" << m_error << std::endl;
}

void EnergySampler::logOutput(const std::vector<double>& params, std::ofstream& outs) {
    const unsigned int prec = 7, width = 16;
    outs << std::scientific << std::setprecision(prec);
    for (unsigned int i = 0; i < m_numberOfParameters; i++) {
        outs << std::setw(width) << params[i] << ",";
    }
    outs << std::scientific << std::setprecision(prec)
        << std::setw(width) << m_energy << ","
        << std::setw(width) << m_variance << ","
        << std::setw(width) << m_error << ","
        << std::setw(width) << m_elapsedTime.count() << ","
        << std::setw(width) << ((double)m_numberOfAcceptedSteps) / ((double)m_numberOfMetropolisSteps)
        << std::endl;
}

void EnergySampler::logOutput(std::ofstream& outs) {
    const unsigned int prec = 10, width = 19;
    outs << std::scientific << std::setprecision(prec)
        << std::setw(width) << m_energy << ","
        << std::setw(width) << m_variance << ","
        << std::setw(width) << m_error << ","
        << std::setw(width) << m_elapsedTime.count() << ","
        << std::setw(width) << ((double)m_numberOfAcceptedSteps) / ((double)m_numberOfMetropolisSteps)
        << std::endl;
}

void EnergySampler::computeAverages() {
    /* Compute the averages of the sampled quantities.
     */
    m_energy = m_cumulativeEnergy / m_numberOfMetropolisSteps;
    m_energySQ = m_cumulativeEnergySQ / m_numberOfMetropolisSteps;
    m_variance = m_energySQ - sq(m_energy);
    m_error = sqrt(m_variance / m_numberOfMetropolisSteps);
    m_elapsedTime = m_watch_end - m_watch_start;
    for (unsigned int i = 0; i < m_numberOfParameters; i++) {
        m_covariance[i] /= m_numberOfMetropolisSteps;   // calculate  <O E>
        m_opO[i] /= m_numberOfMetropolisSteps;          // calculate  <O>
        m_covariance[i] -= m_opO[i] * m_energy;         // subtract  <O> <E>
    }
}

std::vector<double> EnergySampler::get_dEdW() const {
    std::vector<double> dEdW(m_numberOfParameters);
    for (unsigned i = 0; i < m_numberOfParameters; i++) {
        dEdW[i] = 2 * m_covariance[i];
    }
    return dEdW;
}