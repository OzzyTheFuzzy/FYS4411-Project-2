#pragma once
#include <memory>
#include <vector>

class Sampler {
public:
    Sampler(
        unsigned int numberOfParticles,
        unsigned int numberOfDimensions,
        double stepLength,
        unsigned int numberOfMetropolisSteps,
        bool storeEnergyHistory = false,
        bool storeDensityHist = false,
        bool densityOnly = false,
        unsigned int densityBins = 200,
        double zMax = 4.0,
        double rhoMax = 4.0
    );

    void sample(bool acceptedStep, class System* system);
    void printOutputToTerminal(class System& system);
    void computeAverages();

    double getEnergy() const;
    double getAcceptanceRatio() const;

    // Gradient dE/dalpha
    double getEnergyGradient() const;

    const std::vector<double>& getEnergyHistory() const;

    // Density getters
    const std::vector<double>& getDensityZ() const;
    const std::vector<double>& getDensityRho() const;
    const std::vector<double>& getDensityZCenters() const;
    const std::vector<double>& getDensityRhoCenters() const;

private:
    unsigned int m_stepNumber = 0;
    unsigned int m_numberOfMetropolisSteps = 0;
    unsigned int m_numberOfParticles = 0;
    unsigned int m_numberOfDimensions = 0;
    unsigned int m_numberOfAcceptedSteps = 0;

    // Energy averages
    double m_energy = 0;
    double m_cumulativeEnergy = 0;

    // Accumulators needed for the VMC gradient formula:
    // dE/dalpha = 2 ( <E_L O_alpha> - <E_L><O_alpha> )
    double m_cumulativeLogDerivativeAlpha = 0.0;
    double m_cumulativeEnergyLogDerivativeAlpha = 0.0;
    double m_energyGradient = 0.0;

    double m_stepLength = 0;

    bool m_storeEnergyHistory = false;
    std::vector<double> m_energyHistory;

    // Density-only mode
    bool m_storeDensityHist = false;
    bool m_densityOnly = false;
    unsigned int m_densityBins = 200;
    double m_zMax = 4.0;
    double m_rhoMax = 4.0;

    std::vector<double> m_densityZCounts;
    std::vector<double> m_densityRhoCounts;
    std::vector<double> m_densityZ;
    std::vector<double> m_densityRho;
    std::vector<double> m_densityZCenters;
    std::vector<double> m_densityRhoCenters;
};
