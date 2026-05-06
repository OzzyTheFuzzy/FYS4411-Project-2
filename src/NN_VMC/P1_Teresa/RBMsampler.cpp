#include <memory>
#include <iostream>
#include <cmath>
#include <vector>
#include <stdexcept>

#include "system.h"
#include "RBMsampler.h"
#include "particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/boltzmannmachine.h"

using std::cout;
using std::endl;

RBMSampler::RBMSampler(
    unsigned int numberOfParticles,
    unsigned int numberOfDimensions,
    unsigned int numberOfHidden,
    double stepLength,
    unsigned int numberOfMetropolisSteps,
    bool storeEnergyHistory,
    bool storeDensityHist,
    bool densityOnly,
    unsigned int densityBins,
    double zMax,
    double rhoMax
)
{
    m_numberOfParticles = numberOfParticles;
    m_numberOfDimensions = numberOfDimensions;
    m_numberOfHidden = numberOfHidden;
    m_stepLength = stepLength;
    m_numberOfMetropolisSteps = numberOfMetropolisSteps;
    m_storeEnergyHistory = storeEnergyHistory;

    if (m_storeEnergyHistory) {
        m_energyHistory.reserve(numberOfMetropolisSteps);
    }

    m_cumulativeDA.assign(
        m_numberOfParticles,
        std::vector<double>(m_numberOfDimensions, 0.0)
    );
    m_cumulativeDB.assign(m_numberOfHidden, 0.0);
    m_cumulativeDW.assign(
        m_numberOfParticles,
        std::vector<std::vector<double>>(
            m_numberOfDimensions,
            std::vector<double>(m_numberOfHidden, 0.0)
        )
    );

    m_cumulativeEDA = m_cumulativeDA;
    m_cumulativeEDB = m_cumulativeDB;
    m_cumulativeEDW = m_cumulativeDW;

    m_gradientA = m_cumulativeDA;
    m_gradientB = m_cumulativeDB;
    m_gradientW = m_cumulativeDW;

    m_storeDensityHist = storeDensityHist;
    m_densityOnly = densityOnly;
    m_densityBins = densityBins;
    m_zMax = zMax;
    m_rhoMax = rhoMax;

    if (m_storeDensityHist) {
        m_densityZCounts.assign(m_densityBins, 0.0);
        m_densityRhoCounts.assign(m_densityBins, 0.0);
        m_densityZ.assign(m_densityBins, 0.0);
        m_densityRho.assign(m_densityBins, 0.0);
        m_densityZCenters.assign(m_densityBins, 0.0);
        m_densityRhoCenters.assign(m_densityBins, 0.0);

        const double dz = 2.0 * m_zMax / static_cast<double>(m_densityBins);
        const double drho = m_rhoMax / static_cast<double>(m_densityBins);

        for (unsigned int i = 0; i < m_densityBins; ++i) {
            m_densityZCenters[i]   = -m_zMax + (i + 0.5) * dz;
            m_densityRhoCenters[i] = (i + 0.5) * drho;
        }
    }
}

void RBMSampler::sample(bool acceptedStep, System* system)
{
    auto* rbm = dynamic_cast<BoltzmannMachine*>(&system->getWaveFunction());
    if (rbm == nullptr) {
        throw std::logic_error("RBMSampler requires a BoltzmannMachine wave function.");
    }

    const double localEnergy = system->computeLocalEnergy();
    if (m_storeEnergyHistory) {
        m_energyHistory.push_back(localEnergy);
    }

    RBMParameterDerivatives der = rbm->computeParameterDerivatives(system->getParticles());

    m_cumulativeEnergy += localEnergy;

    for (unsigned int p = 0; p < m_numberOfParticles; ++p) {
        for (unsigned int d = 0; d < m_numberOfDimensions; ++d) {
            m_cumulativeDA[p][d]  += der.dA[p][d];
            m_cumulativeEDA[p][d] += localEnergy * der.dA[p][d];

            for (unsigned int h = 0; h < m_numberOfHidden; ++h) {
                m_cumulativeDW[p][d][h]  += der.dW[p][d][h];
                m_cumulativeEDW[p][d][h] += localEnergy * der.dW[p][d][h];
            }
        }
    }

    for (unsigned int h = 0; h < m_numberOfHidden; ++h) {
        m_cumulativeDB[h]  += der.dB[h];
        m_cumulativeEDB[h] += localEnergy * der.dB[h];
    }

    m_stepNumber++;
    m_numberOfAcceptedSteps += acceptedStep;
}

void RBMSampler::computeAverages()
{
    const double M = static_cast<double>(m_stepNumber);
    m_energy = m_cumulativeEnergy / M;

    for (unsigned int p = 0; p < m_numberOfParticles; ++p) {
        for (unsigned int d = 0; d < m_numberOfDimensions; ++d) {
            const double meanO = m_cumulativeDA[p][d] / M;
            const double meanEO = m_cumulativeEDA[p][d] / M;
            m_gradientA[p][d] = 2.0 * (meanEO - m_energy * meanO);

            for (unsigned int h = 0; h < m_numberOfHidden; ++h) {
                const double meanOW = m_cumulativeDW[p][d][h] / M;
                const double meanEOW = m_cumulativeEDW[p][d][h] / M;
                m_gradientW[p][d][h] = 2.0 * (meanEOW - m_energy * meanOW);
            }
        }
    }

    for (unsigned int h = 0; h < m_numberOfHidden; ++h) {
        const double meanO = m_cumulativeDB[h] / M;
        const double meanEO = m_cumulativeEDB[h] / M;
        m_gradientB[h] = 2.0 * (meanEO - m_energy * meanO);
    }
}

void RBMSampler::printOutputToTerminal()
{
    cout << endl;
    cout << "  -- RBM Sampler results -- " << endl;
    cout << " Energy : " << m_energy << endl;
    cout << " Acceptance ratio : "
         << static_cast<double>(m_numberOfAcceptedSteps) / static_cast<double>(m_numberOfMetropolisSteps)
         << endl;
    cout << " Gradient shapes: " << endl;
    cout << "   dE/da : (" << m_numberOfParticles << ", " << m_numberOfDimensions << ")" << endl;
    cout << "   dE/db : (" << m_numberOfHidden << ")" << endl;
    cout << "   dE/dW : (" << m_numberOfParticles << ", "
         << m_numberOfDimensions << ", " << m_numberOfHidden << ")" << endl;
    cout << endl;
}

double RBMSampler::getEnergy() const { return m_energy; }

double RBMSampler::getAcceptanceRatio() const
{
    return static_cast<double>(m_numberOfAcceptedSteps) /
           static_cast<double>(m_numberOfMetropolisSteps);
}

const std::vector<std::vector<double>>& RBMSampler::getGradientA() const { return m_gradientA; }
const std::vector<double>& RBMSampler::getGradientB() const { return m_gradientB; }
const std::vector<std::vector<std::vector<double>>>& RBMSampler::getGradientW() const { return m_gradientW; }
const std::vector<double>& RBMSampler::getEnergyHistory() const { return m_energyHistory; }

const std::vector<double>& RBMSampler::getDensityZ() const {
    return m_densityZ;
}

const std::vector<double>& RBMSampler::getDensityRho() const {
    return m_densityRho;
}

const std::vector<double>& RBMSampler::getDensityZCenters() const {
    return m_densityZCenters;
}

const std::vector<double>& RBMSampler::getDensityRhoCenters() const {
    return m_densityRhoCenters;
}