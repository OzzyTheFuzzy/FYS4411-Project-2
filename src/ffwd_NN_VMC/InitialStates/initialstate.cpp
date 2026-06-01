#include <memory>
#include <iostream>
#include <cassert>

#include "initialstate.h"
#include "../particle.h"
#include "Math/random.h"
#include "../common.h"

using namespace CommonUtils;

std::vector<std::unique_ptr<Particle>> setupRandomUniformInitialState(
    unsigned int numberOfDimensions,
    unsigned int numberOfParticles,
    Random& rng,
    double rep_a
) {
    assert(numberOfDimensions > 0 && numberOfParticles > 0);

    auto particles = std::vector<std::unique_ptr<Particle>>();

    for (unsigned int i = 0; i < numberOfParticles; i++) {
        std::vector<double> position(numberOfDimensions);
        bool tooClose = true;

        do {
            for (unsigned int j = 0; j < numberOfDimensions; j++) {
                position[j] = (rng.nextDouble() - 0.5) * 10.0;
            }
            tooClose = false;
            for (unsigned int k = 0; k < i; k++) {
                double dist = 0;
                for (unsigned int j = 0; j < numberOfDimensions; j++) {
                    dist += sq(position[j] - particles[k]->getPosition()[j]);
                }
                if (sqrt(dist) <= rep_a) {
                    tooClose = true;
                    break;
                }
            }
        } while (tooClose);

        particles.push_back(std::make_unique<Particle>(position));
    }

    return particles;
}