#include <iostream>
#include <memory>
#include <cassert>

#include "system.h"
#include "Samplers/energysampler.h"
#include "Samplers/densitysampler.h"
#include "Samplers/NNsampler.h"
#include "particle.h"
#include "WaveFunctions/wavefunction.h"
#include "Hamiltonians/hamiltonian.h"
#include "InitialStates/initialstate.h"
#include "Solvers/montecarlo.h"


System::System(
    std::unique_ptr<class Hamiltonian> hamiltonian,
    std::unique_ptr<class WaveFunction> waveFunction,
    std::unique_ptr<class MonteCarlo> solver,
    std::vector<std::unique_ptr<class Particle>> particles
) {
    m_numberOfParticles = particles.size();;
    m_numberOfDimensions = particles[0]->getNumberOfDimensions();
    m_hamiltonian = std::move(hamiltonian);
    m_waveFunction = std::move(waveFunction);
    m_solver = std::move(solver);
    m_particles = std::move(particles);
    if (m_solver->hasAnalyticalOption()) {
        m_hamiltonian->set_analytic_ifAvailable(m_solver->get_preferAnalytic());
    }
}

System::System(
    std::unique_ptr<class Hamiltonian> hamiltonian,
    std::unique_ptr<class WaveFunction> waveFunction
) {
    m_numberOfParticles = 0;
    m_numberOfDimensions = 0;
    m_hamiltonian = std::move(hamiltonian);
    m_waveFunction = std::move(waveFunction);
}


unsigned int System::runEquilibrationSteps(double stepParameter,
    unsigned int numberOfEquilibrationSteps) {
    unsigned int acceptedSteps = 0;

    for (unsigned int i = 0; i < numberOfEquilibrationSteps; i++) {
        acceptedSteps += m_solver->step(stepParameter, *m_waveFunction, m_particles);
    }

    return acceptedSteps;
}

std::unique_ptr<class EnergySampler> System::runMetropolisSteps(double stepParameter,
    unsigned int numberOfMetropolisSteps, std::ofstream* energiesOut) {
    auto sampler = std::make_unique<EnergySampler>(
        m_numberOfParticles,
        m_numberOfDimensions,
        m_waveFunction->getNumberOfParameters(),
        stepParameter,
        numberOfMetropolisSteps);

    for (unsigned int i = 0; i < numberOfMetropolisSteps; i++) {
        /* Call solver method to do a single Monte-Carlo step.
         */
        bool acceptedStep = m_solver->step(stepParameter, *m_waveFunction, m_particles);

        // Sample energy
        sampler->sample(acceptedStep, this, energiesOut);
    }

    sampler->computeAverages();

    return sampler;
}


// std::unique_ptr<NNsampler> System::runMetropolisSteps_NN(double stepParameter,
//     unsigned int numberOfMetropolisSteps, WaveFunction& wf_train) {
//     std::unique_ptr<NNsampler> sampler = std::make_unique<NNsampler>(
//         m_numberOfParticles,
//         m_numberOfDimensions,
//         m_waveFunction->getNumberOfParameters(),
//         numberOfMetropolisSteps,
//         wf_train);

//     for (unsigned int i = 0; i < numberOfMetropolisSteps; i++) {
//         /* Call solver method to do a single Monte-Carlo step.
//          */
//         bool acceptedStep = m_solver->step(stepParameter, *m_waveFunction, m_particles);

//         // Sample energy
//         sampler->sample_train(acceptedStep, this);
//     }

//     sampler->computeAverages();

//     return sampler;
// }

std::unique_ptr<NNsampler> System::runMetropolisSteps_NN_pretrain(double stepParameter,
    unsigned int numberOfMetropolisSteps, WaveFunction& wf_train) {
    std::unique_ptr<NNsampler> sampler = std::make_unique<NNsampler>(
        m_numberOfParticles,
        m_numberOfDimensions,
        m_waveFunction->getNumberOfParameters(),
        stepParameter,
        numberOfMetropolisSteps,
        wf_train);

    for (unsigned int i = 0; i < numberOfMetropolisSteps; i++) {
        /* Call solver method to do a single Monte-Carlo step.
         */
        bool acceptedStep = m_solver->step(stepParameter, wf_train, m_particles);

        // Sample 
        sampler->sample(acceptedStep, this);
    }

    sampler->computeAverages();

    return sampler;
}

std::unique_ptr<class DensitySampler> System::runMetropolisStepsOnebodyDensity(double stepParameter,
    unsigned int numberOfMetropolisSteps, double rMax, unsigned int nBins) {
    auto sampler = std::make_unique<DensitySampler>(
        m_numberOfParticles,
        m_numberOfDimensions,
        m_waveFunction->getNumberOfParameters(),
        stepParameter,
        numberOfMetropolisSteps,
        rMax,
        nBins);

    for (unsigned int i = 0; i < numberOfMetropolisSteps; i++) {
        /* Call solver method to do a single Monte-Carlo step.
         */
        bool acceptedStep = m_solver->step(stepParameter, *m_waveFunction, m_particles);

        // sample 1
        sampler->sample(acceptedStep, this);
    }

    sampler->computeAverages();

    return sampler;
}

double System::computeLocalEnergy() {
    if (m_solver->get_useCache()) {
        return m_hamiltonian->computeLocalEnergy(
            *m_waveFunction, m_particles, m_solver->getCache());
    }
    else {
        return m_hamiltonian->computeLocalEnergy(
            *m_waveFunction, m_particles);    
    }
}

double System::computeParamDerivativeLn(unsigned int param_idx) {
    // Helper function
    return m_waveFunction->computeParamDerivativeLn(m_particles, param_idx);
}

std::vector<double> System::computeLogParDer_vect() {
    // Helper function
    return m_waveFunction->computeLogParDer_vect(m_particles);
}

const std::vector<double>& System::getWaveFunctionParameters() {
    // Helper function
    return m_waveFunction->getParameters();
}

Hamiltonian& System::getHamiltonian() {
    return *m_hamiltonian;
}

WaveFunction& System::getWaveFunction() {
    return *m_waveFunction;
}

void System::setParticles(std::vector<std::unique_ptr<Particle>> new_particles) {
    m_particles = std::move(new_particles);
    m_numberOfParticles = m_particles.size();
    m_numberOfDimensions = m_particles[0]->getNumberOfDimensions();
}

void System::setSolver(std::unique_ptr<MonteCarlo> new_solver) {
    m_solver = std::move(new_solver);
    if (m_solver->hasAnalyticalOption()) {
        m_hamiltonian->set_analytic_ifAvailable(m_solver->get_preferAnalytic());
    }
}

void System::setHamiltonian(std::unique_ptr<Hamiltonian> new_hamiltonian) {
    m_hamiltonian = std::move(new_hamiltonian);
}

std::unique_ptr<WaveFunction> System::setWaveFunction(std::unique_ptr<WaveFunction> new_waveFunction) {
    auto temp = std::move(m_waveFunction);
    m_waveFunction = std::move(new_waveFunction);
    return temp;
}
