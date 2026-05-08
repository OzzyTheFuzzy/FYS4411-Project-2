#include <memory>
#include <iostream>
#include <cmath>
#include <vector>
#include <stdexcept>

#include "system.h"
#include "NNsampler.h"
#include "particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/nn_envelope.h"
#include "WaveFunctions/wavefunction.h"

using std::cout;
using std::endl;

NNsampler::NNsampler(
    unsigned int numberOfParticles,
    unsigned int numberOfDimensions,
    unsigned int numberOfParameters,
    double stepLength,
    unsigned int numberOfMetropolisSteps,
    WaveFunction& wf_train
) : m_numberOfParticles(numberOfParticles),
m_numberOfDimensions(numberOfDimensions),
m_numberOfParameters(numberOfParameters),
m_stepLength(stepLength),
m_numberOfMetropolisSteps(numberOfMetropolisSteps),
m_wf_train(wf_train) {

    if (m_storeEnergyHistory) {
        m_energyHistory.reserve(numberOfMetropolisSteps);
    }

    m_cumulativeOW.assign(m_numberOfParameters, 0);
    m_OW.assign(m_numberOfParameters, 0);
    m_cumulativeAOW.assign(m_numberOfParameters, 0);
    m_cumulativeEOW.assign(m_numberOfParameters, 0);
}

void NNsampler::sample(bool acceptedStep, System* system) {
    auto* rbm = dynamic_cast<NN_envelope*>(&system->getWaveFunction());
    if (rbm == nullptr) {
        throw std::logic_error("NNsampler requires a NN_envelope wave function.");
    }

    const double localEnergy = system->computeLocalEnergy();
    if (m_storeEnergyHistory) {
        m_energyHistory.push_back(localEnergy);
    }
    auto OW = system->getWaveFunction().computeLogParDer(system->getParticles());

    m_cumulativeEnergy += localEnergy;

    auto& particles = system->getParticles();
    double psi = system->getWaveFunction().evaluate(particles);
    double psi_train = m_wf_train.evaluate(particles);
    m_cumulativeA += psi_train / psi;
    m_cumulativeA2 += (psi_train / psi) * (psi_train / psi);
    for (int i = 0; i < m_numberOfParameters; i++) {
        m_cumulativeOW[i] += OW[i];
        m_cumulativeAOW[i] += psi_train / psi * OW[i];
        m_cumulativeEOW[i] += localEnergy * OW[i];
    }

    m_stepNumber++;
    m_numberOfAcceptedSteps += acceptedStep;
}

void NNsampler::computeAverages() {
    // const double M = static_cast<double>(m_stepNumber);
    m_energy = m_cumulativeEnergy / (double) m_stepNumber;
    m_A = m_cumulativeA / (double) m_stepNumber;
    m_A2 = m_cumulativeA2 / (double) m_stepNumber;
    for (int i = 0; i < m_numberOfParameters; i++) {
        m_OW[i] = m_cumulativeOW[i] / (double) m_stepNumber;
        m_AOW[i] = m_cumulativeAOW[i] / (double) m_stepNumber;
        m_EOW[i] = m_cumulativeEOW[i] / (double) m_stepNumber;
    }
    m_K = m_A * m_A / m_A2;
}

void NNsampler::printOutputToTerminal() {
    cout << endl;
    cout << "  -- NN Sampler results -- " << endl;
    cout << " Energy : " << m_energy << endl;
    cout << " K : " << m_K << endl;
    // cout << " dEdW : " << m_energy << endl;
    cout << " Acceptance ratio : "
        << (double) m_numberOfAcceptedSteps / (double) m_numberOfMetropolisSteps
        << endl;
    cout << endl;
}

double NNsampler::getEnergy() const { return m_energy; }

double NNsampler::getAcceptanceRatio() const {
    return (double) m_numberOfAcceptedSteps / (double) m_numberOfMetropolisSteps;
}

std::vector<double> NNsampler::get_dEdW() const {
    std::vector<double> dEdW(m_numberOfParameters);
    for (int i = 0; i < m_numberOfParameters; i++) {
        dEdW[i] = 2 * (m_EOW[i] - m_energy * m_OW[i]);
    }
    return dEdW;
}

std::vector<double> NNsampler::get_dKdW() const {
    std::vector<double> dKdW(m_numberOfParameters);
    for (int i = 0; i < m_numberOfParameters; i++) {
        dKdW[i] = 2 * m_K * (m_AOW[i] / m_A - m_OW[i]);
    }
    return dKdW;
}