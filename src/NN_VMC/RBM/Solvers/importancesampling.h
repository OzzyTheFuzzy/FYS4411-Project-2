#pragma once
#include <memory>
#include "montecarlo.h"

class ImportanceSampling : public MonteCarlo {
public:
    ImportanceSampling(std::unique_ptr<class Random> rng, double diffusionConstant = 0.5);
    bool step(
        double timeStep,
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles
    ) override;

private:
    double m_diffusion = 0.5;
};