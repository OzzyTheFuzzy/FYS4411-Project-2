#include <iostream>
#include <memory>
#include <cassert>

#include "system.h"
#include "sampler.h"
#include "RBMsampler.h"
#include "particle.h"
#include "WaveFunctions/wavefunction.h"
#include "Hamiltonians/hamiltonian.h"
#include "InitialStates/initialstate.h"
#include "Solvers/montecarlo.h"


System::System(
        std::unique_ptr<class Hamiltonian> hamiltonian,
        std::unique_ptr<class WaveFunction> waveFunction,
        std::unique_ptr<class MonteCarlo> solver,
        std::vector<std::unique_ptr<class Particle>> particles)
{
    m_numberOfParticles = particles.size();;
    m_numberOfDimensions = particles[0]->getNumberOfDimensions();
    m_hamiltonian = std::move(hamiltonian);
    m_waveFunction = std::move(waveFunction);
    m_solver = std::move(solver);
    m_particles = std::move(particles);
}


unsigned int System::runEquilibrationSteps(
        double stepLength,
        unsigned int numberOfEquilibrationSteps)
{
    unsigned int acceptedSteps = 0;

    for (unsigned int i = 0; i < numberOfEquilibrationSteps; i++) {
        acceptedSteps += m_solver->step(stepLength, *m_waveFunction, m_particles);
    }

    return acceptedSteps;
}

std::unique_ptr<class Sampler> System::runMetropolisSteps(
        double stepLength,
        unsigned int numberOfMetropolisSteps,
        bool storeEnergyHistory,
        bool storeDensityHist,
        bool densityOnly,
        unsigned int densityBins,
        double densityZMax,
        double densityRhoMax)
{
    auto sampler = std::make_unique<Sampler>(
            m_numberOfParticles,
            m_numberOfDimensions,
            stepLength,
            numberOfMetropolisSteps,
            storeEnergyHistory,
            storeDensityHist,
            densityOnly,
            densityBins,
            densityZMax,
            densityRhoMax);

    for (unsigned int i = 0; i < numberOfMetropolisSteps; i++) {
        // Do one Monte Carlo step
        bool acceptedStep = m_solver->step(stepLength, *m_waveFunction, m_particles);

        // Sample observables
        sampler->sample(acceptedStep, this);
    }

    sampler->computeAverages();

    return sampler;
}

std::unique_ptr<class RBMSampler> System::runRBMMetropolisSteps(
    double stepLength,
    unsigned int numberOfMetropolisSteps,
    unsigned int numberOfHidden,
    bool storeEnergyHistory
)
{
    auto sampler = std::make_unique<RBMSampler>(
        m_numberOfParticles,
        m_numberOfDimensions,
        numberOfHidden,
        stepLength,
        numberOfMetropolisSteps,
        storeEnergyHistory
    );

    for (unsigned int i = 0; i < numberOfMetropolisSteps; i++) {
        bool acceptedStep = m_solver->step(stepLength, *m_waveFunction, m_particles);
        sampler->sample(acceptedStep, this);
    }

    sampler->computeAverages();
    return sampler;
}
double System::computeLocalEnergy()
{
    // Helper function
    return m_hamiltonian->computeLocalEnergy(*m_waveFunction, m_particles);
}

const std::vector<double>& System::getWaveFunctionParameters()
{
    // Helper function
    return m_waveFunction->getParameters();
}

WaveFunction& System::getWaveFunction()
{
    return *m_waveFunction;
}

std::vector<std::unique_ptr<class Particle>>& System::getParticles()
{
    return m_particles;
}