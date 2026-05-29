#include <memory>
#include <iostream>
#include <cassert>

#include "initialstate.h"
#include "../particle.h"
#include "Math/random.h"


std::vector<std::unique_ptr<Particle>> setupRandomUniformInitialState(
    double stepLength,
    unsigned int numberOfDimensions,
    unsigned int numberOfParticles,
    Random& rng
)
{
    assert(numberOfDimensions > 0 && numberOfParticles > 0);

    auto particles = std::vector<std::unique_ptr<Particle>>();

    for (unsigned int i=0; i < numberOfParticles; i++) {
        std::vector<double> position = std::vector<double>();

        for (unsigned int j=0; j < numberOfDimensions; j++) {
            double x = (rng.nextDouble() * 2.0 - 1.0) * stepLength;
            position.push_back(x);
        }

        particles.push_back(std::make_unique<Particle>(position));
    }

    return particles;
}

static double distanceSquared(
    const std::vector<double>& a,
    const std::vector<double>& b
)
{
    double r2 = 0.0;
    for (unsigned int d = 0; d < a.size(); ++d) {
        const double diff = a[d] - b[d];
        r2 += diff * diff;
    }
    return r2;
}

std::vector<std::unique_ptr<Particle>> setupRandomUniformInitialStateNoOverlap(
    double stepLength,
    unsigned int numberOfDimensions,
    unsigned int numberOfParticles,
    double hardCoreDiameter,
    Random& rng,
    unsigned int maxTriesPerParticle
)
{
    assert(numberOfDimensions > 0 && numberOfParticles > 0);
    assert(hardCoreDiameter >= 0.0);

    auto particles = std::vector<std::unique_ptr<Particle>>();
    const double a2 = hardCoreDiameter * hardCoreDiameter;

    for (unsigned int i = 0; i < numberOfParticles; ++i) {
        bool placed = false;

        for (unsigned int attempt = 0; attempt < maxTriesPerParticle; ++attempt) {
            std::vector<double> position(numberOfDimensions, 0.0);
            for (unsigned int d = 0; d < numberOfDimensions; ++d) {
                position[d] = (rng.nextDouble() * 2.0 - 1.0) * stepLength;
            }

            bool overlap = false;
            for (const auto& p : particles) {
                if (distanceSquared(position, p->getPosition()) <= a2) {
                    overlap = true;
                    break;
                }
            }

            if (!overlap) {
                particles.push_back(std::make_unique<Particle>(position));
                placed = true;
                break;
            }
        }

        if (!placed) {
            std::cerr << "Error: could not place particle " << i
                      << " without overlap. Try a larger init range.\n";
            std::exit(1);
        }
    }

    return particles;
}