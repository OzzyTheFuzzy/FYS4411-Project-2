#pragma once
#include <memory>
#include <vector>
#include <chrono>

/**
 * @brief Abstract base class for all Monte Carlo samplers.
 * * Defines the interface for collecting data during a VMC run.
 * Derived classes (like EnergySampler or DensitySampler) will implement
 * specific measurements (e.g., local energy, radial density).
 */
class Sampler {
public:
    /**
     * @brief Constructs the base sampler and initializes counters.
     * @param numberOfParticles Total number of particles in the system.
     * @param numberOfDimensions Spatial dimensions.
     * @param numberOfParameters Number of variational parameters.
     * @param stepLength The step size used in the Metropolis algorithm.
     * @param numberOfMetropolisSteps Total number of steps to be executed.
     */
    Sampler(unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        unsigned int numberOfParameters,
        double stepLength,
        unsigned int numberOfMetropolisSteps);
    virtual ~Sampler() = default;

    /**
     * @brief Samples the system's properties at the current Metropolis step.
     * @param acceptedStep True if the proposed Metropolis move was accepted.
     * @param system Pointer to the quantum system to extract properties from.
     * @param outfile Optional output stream to log raw step-by-step data.
     */
    virtual void sample(bool acceptedStep, class System* system, std::ofstream* outfile = nullptr) = 0;

    /**
     * @brief Computes final statistical averages (e.g., means, variances, errors).
     * * This should be called exactly once after the Metropolis loop finishes.
     */
    virtual void computeAverages() = 0;

protected:
    unsigned int m_stepNumber = 0;
    unsigned int m_numberOfMetropolisSteps = 0;
    unsigned int m_numberOfParticles = 0;
    unsigned int m_numberOfDimensions = 0;
    unsigned int m_numberOfParameters = 0;
    unsigned int m_numberOfAcceptedSteps = 0;
    double m_stepLength = 0;
    std::chrono::high_resolution_clock::time_point m_watch_start;
    std::chrono::high_resolution_clock::time_point m_watch_end;
    std::chrono::duration<double> m_elapsedTime;
};
