#include <memory>
#include <iostream>
#include <cmath>
#include <vector>
#include "system.h"
#include "sampler.h"
#include "particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"

using std::cout;
using std::endl;


Sampler::Sampler(
        unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        double stepLength,
        unsigned int numberOfMetropolisSteps,
        bool storeEnergyHistory,
        bool storeDensityHist,
        bool densityOnly,
        unsigned int densityBins,
        double zMax,
        double rhoMax)
{
    m_stepNumber = 0;
    m_numberOfMetropolisSteps = numberOfMetropolisSteps;
    m_numberOfParticles = numberOfParticles;
    m_numberOfDimensions = numberOfDimensions;
    m_energy = 0;
    m_cumulativeEnergy = 0;

    m_cumulativeLogDerivativeAlpha = 0.0;
    m_cumulativeEnergyLogDerivativeAlpha = 0.0;
    m_energyGradient = 0.0;

    m_stepLength = stepLength;
    m_numberOfAcceptedSteps = 0;

    m_storeEnergyHistory = storeEnergyHistory;
    if (m_storeEnergyHistory) {
        m_energyHistory.reserve(numberOfMetropolisSteps);
    }

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

void Sampler::sample(bool acceptedStep, System* system) {
    // 1) Density histograms
    if (m_storeDensityHist) {
        const double dz = 2.0 * m_zMax / static_cast<double>(m_densityBins);
        const double drho = m_rhoMax / static_cast<double>(m_densityBins);

        auto& particles = system->getParticles();
        for (auto& p : particles) {
            const auto& r = p->getPosition();

            const double z = r[2];
            const double rho = std::sqrt(r[0]*r[0] + r[1]*r[1]);

            int iz = static_cast<int>((z + m_zMax) / dz);
            int irho = static_cast<int>(rho / drho);

            if (iz >= 0 && iz < static_cast<int>(m_densityBins)) {
                m_densityZCounts[iz] += 1.0;
            }
            if (irho >= 0 && irho < static_cast<int>(m_densityBins)) {
                m_densityRhoCounts[irho] += 1.0;
            }
        }
    }

    // 2) Skip energy work in density-only mode
    if (!m_densityOnly) {
        double localEnergy = system->computeLocalEnergy();
        if (m_storeEnergyHistory) {
            m_energyHistory.push_back(localEnergy);
        }

        double logDerivativeAlpha =
            system->getWaveFunction().computeLogDerivativeAlpha(system->getParticles());

        m_cumulativeEnergy += localEnergy;
        m_cumulativeLogDerivativeAlpha += logDerivativeAlpha;
        m_cumulativeEnergyLogDerivativeAlpha += localEnergy * logDerivativeAlpha;
    }

    m_stepNumber++;
    m_numberOfAcceptedSteps += acceptedStep;
}

const std::vector<double>& Sampler::getEnergyHistory() const {
    return m_energyHistory;
}

void Sampler::printOutputToTerminal(System& system) {
    auto pa = system.getWaveFunctionParameters();
    auto p = pa.size();

    cout << endl;
    cout << "  -- System info -- " << endl;
    cout << " Number of particles  : " << m_numberOfParticles << endl;
    cout << " Number of dimensions : " << m_numberOfDimensions << endl;
    cout << " Number of Metropolis steps run : 10^" << std::log10(m_numberOfMetropolisSteps) << endl;
    cout << " Step length used : " << m_stepLength << endl;
    cout << " Ratio of accepted steps: " 
         << static_cast<double>(m_numberOfAcceptedSteps) / static_cast<double> (m_numberOfMetropolisSteps) << endl;
    cout << endl;
    cout << "  -- Wave function parameters -- " << endl;
    cout << " Number of parameters : " << p << endl;
    for (unsigned int i=0; i < p; i++) {
        cout << " Parameter " << i+1 << " : " << pa.at(i) << endl;
    }
    cout << endl;
    cout << "  -- Results -- " << endl;
    cout << " Energy : " << m_energy << endl;
    cout << "dE/dalpha : " << m_energyGradient << endl;
    cout << endl;
}

void Sampler::computeAverages() {
    double M = static_cast<double>(m_stepNumber);

    if (!m_densityOnly) {
        m_energy = m_cumulativeEnergy / M;

        double meanLogDerivativeAlpha =
            m_cumulativeLogDerivativeAlpha / M;

        double meanEnergyLogDerivativeAlpha =
            m_cumulativeEnergyLogDerivativeAlpha / M;

        m_energyGradient =
            2.0 * (meanEnergyLogDerivativeAlpha - m_energy * meanLogDerivativeAlpha);
    } else {
        m_energy = 0.0;
        m_energyGradient = 0.0;
    }

    if (m_storeDensityHist) {
        const double dz = 2.0 * m_zMax / static_cast<double>(m_densityBins);
        const double drho = m_rhoMax / static_cast<double>(m_densityBins);

        const double norm = M * static_cast<double>(m_numberOfParticles);

        for (unsigned int i = 0; i < m_densityBins; ++i) {
            m_densityZ[i] = m_densityZCounts[i] / (norm * dz);
            m_densityRho[i] = m_densityRhoCounts[i] / (norm * drho);
        }
    }
}

double Sampler::getEnergy() const {
    return m_energy;
}

double Sampler::getAcceptanceRatio() const {
    return static_cast<double>(m_numberOfAcceptedSteps) /
           static_cast<double>(m_numberOfMetropolisSteps);
}

double Sampler::getEnergyGradient() const {
    return m_energyGradient;
}

const std::vector<double>& Sampler::getDensityZ() const {
    return m_densityZ;
}

const std::vector<double>& Sampler::getDensityRho() const {
    return m_densityRho;
}

const std::vector<double>& Sampler::getDensityZCenters() const {
    return m_densityZCenters;
}

const std::vector<double>& Sampler::getDensityRhoCenters() const {
    return m_densityRhoCenters;
}