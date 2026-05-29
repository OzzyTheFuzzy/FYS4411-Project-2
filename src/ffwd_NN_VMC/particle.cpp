#include "particle.h"
#include <cassert>

Particle::Particle(const std::vector<double>& position) {
    m_numberOfDimensions = position.size();
    m_position = position;
}

void Particle::adjustPosition(double change, unsigned int dimension) {
    m_position[dimension] += change;
}

void Particle::setPosition(double value, unsigned int dimension) {
    m_position[dimension] = value;
}

void Particle::setPosition(const std::vector<double> loc) {
    m_position = loc;
}