#include "utilities.h"

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <limits>
#include <cstdlib>
#include <cmath>

#include "system.h"
#include "WaveFunctions/simplegaussian.h"
#include "WaveFunctions/correlatedgaussian.h"
#include "WaveFunctions/anisotropicgaussian.h"
#include "Hamiltonians/harmonicoscillator.h"
#include "Hamiltonians/interactingelliptictrap.h"
#include "InitialStates/initialstate.h"
#include "Solvers/metropolis.h"
#include "Solvers/importancesampling.h"
#include "Math/random.h"
#include "particle.h"
#include "sampler.h"
#include "Solvers/montecarlo.h"

#ifdef _OPENMP
#include <omp.h>
#endif

std::string modeToString(Mode m) {
    return (m == Mode::Importance) ? "is" : "bf";
}
std::string evalToString(EvalType e) {
    return (e == EvalType::Numerical) ? "nu" : "an";
}

// turn 0.005 -> "0p005" for filenames
std::string doubleToTag(double x) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << x;
    std::string s = oss.str();
    for (char& c : s) if (c == '.') c = 'p';
    // trim trailing zeros
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == 'p') s.push_back('0');
    return s;
}

RunResult runVMC(
    unsigned int N,
    unsigned int D,
    unsigned int nMetropolis,
    unsigned int nEquil,
    double omega,
    double alpha,
    double stepParam,            // bf: stepLength,  is: dt
    Mode mode,
    int seed,
    bool useNumericalLaplacian,  // affects local energy evaluation only
    double hFD,                  // finite diff step used by HarmonicOscillator if numerical Laplacian is enabled
    bool printToTerminal,
    bool storeEnergyHistory,
    bool useInteraction,
    double beta,
    double gamma,
    double hardCoreA,
    bool densityOnly,
    bool storeDensityHist,
    bool useNoJastrow,
    unsigned int densityBins,
    double densityZMax,
    double densityRhoMax
) {
    // For initialization we do NOT want to use dt (too small),
    // so we use a fixed init range.
    const double initRange = 1.0;

    if (useInteraction) {
        if (D != 3) {
            std::cerr << "Error: interacting hard-sphere implementation is currently only available for D=3.\n";
            std::exit(1);
        }
        if (mode == Mode::Importance) {
            std::cerr << "Error: interacting case is not yet implemented for importance sampling.\n";
            std::exit(1);
        }
        if (useNumericalLaplacian) {
            std::cerr << "Error: interacting case should use the analytic local-energy formulas, not numerical Laplacian.\n";
            std::exit(1);
        }
    }

    auto rng = std::make_unique<Random>(seed);

    std::vector<std::unique_ptr<Particle>> particles;
    if (useInteraction) {
        particles = setupRandomUniformInitialStateNoOverlap(
            initRange, D, N, hardCoreA, *rng
        );
    } else {
        particles = setupRandomUniformInitialState(
            initRange, D, N, *rng
        );
    }

    std::unique_ptr<MonteCarlo> solver;
    if (mode == Mode::Importance) {
        solver = std::make_unique<ImportanceSampling>(std::move(rng), 0.5);
    } else {
        solver = std::make_unique<Metropolis>(std::move(rng));
    }

    std::unique_ptr<Hamiltonian> hamiltonian;
    std::unique_ptr<WaveFunction> waveFunction;

    if (useInteraction) {
        hamiltonian = std::make_unique<InteractingEllipticTrap>(gamma);

        if (useNoJastrow) {
            waveFunction = std::make_unique<AnisotropicGaussian>(alpha, beta);
        } else {
            waveFunction = std::make_unique<CorrelatedGaussian>(alpha, beta, hardCoreA);
        }
    } else {
        hamiltonian = std::make_unique<HarmonicOscillator>(omega, useNumericalLaplacian, hFD);
        waveFunction = std::make_unique<SimpleGaussian>(alpha);
    }

    auto system = std::make_unique<System>(
        std::move(hamiltonian),
        std::move(waveFunction),
        std::move(solver),
        std::move(particles)
    );

    system->runEquilibrationSteps(stepParam, nEquil);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto sampler = system->runMetropolisSteps(
        stepParam, 
        nMetropolis, 
        storeEnergyHistory,
        storeDensityHist,
        densityOnly,
        densityBins,
        densityZMax,
        densityRhoMax
    );
    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> dt = t1 - t0;

    RunResult res;
    res.cpu_seconds = dt.count();
    res.energy = sampler->getEnergy();
    res.gradient = sampler -> getEnergyGradient();
    res.acceptance = sampler->getAcceptanceRatio();

    if (storeDensityHist) {
        res.densityZ = sampler->getDensityZ();
        res.densityRho = sampler->getDensityRho();
        res.densityZCenters = sampler->getDensityZCenters();
        res.densityRhoCenters = sampler->getDensityRhoCenters();
    }

    // Store the energy values
    if (storeEnergyHistory){
        res.energyHistory = sampler ->getEnergyHistory();
    }

    if (printToTerminal) {
        std::cout << "CPU time (production run): " << res.cpu_seconds << " s\n";
        sampler->printOutputToTerminal(*system);
    }

    return res;
}

ParallelRunResult runVMCReplicasParallel(
    unsigned int N,
    unsigned int D,
    unsigned int nMetropolis,
    unsigned int nEquil,
    double omega,
    double alpha,
    double stepParam,
    Mode mode,
    int baseSeed,
    bool useNumericalLaplacian,
    double hFD,
    int nReplicas,
    bool storeEnergyHistory,
    bool useInteraction,
    double beta,
    double gamma,
    double hardCoreA 
) {
    if (nReplicas < 1) {
        nReplicas = 1;
    }

    ParallelRunResult out;
    out.nReplicas = nReplicas;
    out.replicas.resize(nReplicas);

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < nReplicas; ++r) {
        // Large stride to avoid accidental overlap patterns in seeds
        int seed = baseSeed + 1000003 * r;

        out.replicas[r] = runVMC(
            N, D,
            nMetropolis,
            nEquil,
            omega,
            alpha,
            stepParam,
            mode,
            seed,
            useNumericalLaplacian,
            hFD,
            false,   // avoid mixed terminal output from many threads
            storeEnergyHistory,
            useInteraction,
            beta,
            gamma,
            hardCoreA
        );
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    out.wall_seconds = dt.count();

    // Average the replica results
    for (const auto& r : out.replicas) {
        out.mean.energy += r.energy;
        out.mean.gradient += r.gradient;
        out.mean.acceptance += r.acceptance;
        out.mean.cpu_seconds += r.cpu_seconds;
    }

    out.mean.energy /= static_cast<double>(nReplicas);
    out.mean.gradient /= static_cast<double>(nReplicas);
    out.mean.acceptance /= static_cast<double>(nReplicas);
    out.mean.cpu_seconds /= static_cast<double>(nReplicas);

    // Spread across replica mean energies
    if (nReplicas > 1) {
        double accum = 0.0;
        for (const auto& r : out.replicas) {
            double diff = r.energy - out.mean.energy;
            accum += diff * diff;
        }
        out.energy_std = std::sqrt(accum / static_cast<double>(nReplicas - 1));
        out.energy_stderr = out.energy_std / std::sqrt(static_cast<double>(nReplicas));
    }

    return out;
}

std::pair<double,double> runAlphaScan(
    unsigned int N,
    unsigned int D,
    unsigned int scanSteps,
    unsigned int scanEquil,
    double omega,
    double alphaMin,
    double alphaMax,
    int nAlpha,
    double stepParam,
    Mode mode,
    int baseSeed,
    double hFD,
    const std::string& scanFile,
    bool useInteraction,
    double beta,
    double gamma,
    double hardCoreA
) {
    std::ofstream outScan(scanFile);
    if (!outScan) {
        std::cerr << "Error: could not open " << scanFile << "\n";
        std::exit(1);
    }
    outScan << "# alpha  energy  cpu_seconds  acceptance\n";
    outScan << std::setprecision(10);

    double bestAlpha = alphaMin;
    double bestEnergy = std::numeric_limits<double>::infinity();

    for (int i = 0; i < nAlpha; i++) {
        const double alpha = alphaMin + i * (alphaMax - alphaMin) / (nAlpha - 1);

        const int seed = baseSeed
            + 1000 * static_cast<int>(N)
            + 10 * static_cast<int>(D)
            + i;

        // Scan ALWAYS uses analytic local energy (fast & stable)
        RunResult r = runVMC(
            N, D,
            scanSteps,
            scanEquil,
            omega,
            alpha,
            stepParam,
            mode,
            seed,
            false, // analytic Laplacian
            hFD,
            false,
            false,
            useInteraction,
            beta,
            gamma,
            hardCoreA
        );

        outScan << alpha << " " << r.energy << " " << r.cpu_seconds << " " << r.acceptance << "\n";

        if (r.energy < bestEnergy) {
            bestEnergy = r.energy;
            bestAlpha = alpha;
        }
    }
    outScan.close();
    return {bestAlpha, bestEnergy};
}