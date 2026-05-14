#include <memory>
#include <cmath>
#include <cassert>
#include <iostream>

#include "../common.h"
#include "wavefunctioncache.h"
#include "../system.h"
#include "../particle.h"

using namespace CommonUtils;

WaveFunctionCache::WaveFunctionCache(WaveFunction& waveFunction,
    std::vector<std::unique_ptr<class Particle>>& particles)
    : m_wf(waveFunction) {
    unsigned int N = particles.size();
    m_particleLn.resize(N);
    for (unsigned int i = 0; i < N; i++) {
        m_particleLn[i] = m_wf.computeParticleLn(particles, i);
    }
}

double WaveFunctionCache::computeLnRatio(std::vector<std::unique_ptr<class Particle>>& particles,
    unsigned int particle_idx) {
    m_particleToUpd = particle_idx;
    m_pendingLn = m_wf.computeParticleLn(particles, particle_idx);
    return m_pendingLn - m_particleLn[particle_idx];
}

void WaveFunctionCache::acceptMove(std::vector<std::unique_ptr<class Particle>>& particles) {
    // Update the moved particle
    m_particleLn[m_particleToUpd] = m_pendingLn;
 
    // All other particles' cached ln values include Jastrow terms u(r_pk)
    // that depend on particle_idx's position
    for (unsigned int k = 0; k < m_particleLn.size(); k++) {
        if (k == m_particleToUpd) continue;
        m_particleLn[k] = m_wf.computeParticleLn(particles, k);
    }
}

double WaveFunctionCache::computeNumericalLaplacian(
    std::vector<std::unique_ptr<Particle>>& particles,
    WaveFunction& waveFunction) {
    double sum = 0.0;
    unsigned int ndim = particles[0]->getNumberOfDimensions();

    for (unsigned int p = 0; p < particles.size(); p++) {
        double cachedLn = m_particleLn[p];  // ln(psi_p) before +-h

        for (unsigned int d = 0; d < ndim; d++) {
            double h = 1e-3;
            // double h = 1e-4 * std::max(1.0, std::abs(particles[p]->getPosition()[d]));

            // +h on particle p
            particles[p]->adjustPosition(h, d);
            double lnPlus = waveFunction.computeParticleLn(particles, p);

            // -2h on particle p
            particles[p]->adjustPosition(-2.0 * h, d);
            double lnMinus = waveFunction.computeParticleLn(particles, p);

            // +h on particle p (restore)
            particles[p]->adjustPosition(h, d);

            // Laplacian in log space:
            // (d2 f) / f = d2 ln f + (d ln f)2
            double d2Ln = (lnPlus - 2.0 * cachedLn + lnMinus) / sq(h);
            double dLn = (lnPlus - lnMinus) / (2.0 * h);

            sum += d2Ln + sq(dLn);
        }
    }

    return sum;
}