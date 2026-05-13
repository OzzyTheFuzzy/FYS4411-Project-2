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
#include <stdexcept>
#include <algorithm>

#include "system.h"
#include "WaveFunctions/simplegaussian.h"
#include "WaveFunctions/correlatedgaussian.h"
#include "WaveFunctions/anisotropicgaussian.h"
#include "Hamiltonians/elliptictrap.h"
#include "Hamiltonians/rbminteractrap.h"
#include "InitialStates/initialstate.h"
#include "Solvers/metropolis.h"
#include "Solvers/importancesampling.h"
#include "Math/random.h"
#include "particle.h"
#include "sampler.h"
#include "RBMsampler.h"
#include "Solvers/montecarlo.h"

#ifdef _OPENMP
#include <omp.h>
#endif

std::string modeToString(Mode m) {
    return (m == Mode::Importance) ? "is" : "bf";
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

namespace {
// ---------- shape checks ----------
bool sameShape2D(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B
) {
    if (A.size() != B.size()) return false;
    for (std::size_t i = 0; i < A.size(); ++i) {
        if (A[i].size() != B[i].size()) return false;
    }
    return true;
}
bool sameShape3D(
    const std::vector<std::vector<std::vector<double>>>& A,
    const std::vector<std::vector<std::vector<double>>>& B
) {
    if (A.size() != B.size()) return false;
    for (std::size_t i = 0; i < A.size(); ++i) {
        if (A[i].size() != B[i].size()) return false;
        for (std::size_t j = 0; j < A[i].size(); ++j) {
            if (A[i][j].size() != B[i][j].size()) return false;
        }
    }
    return true;
}
// ---------- zero-like constructors ----------
std::vector<double> zeroLike1D(const std::vector<double>& A) {
    return std::vector<double>(A.size(), 0.0);
}
std::vector<std::vector<double>> zeroLike2D(
    const std::vector<std::vector<double>>& A
) {
    std::vector<std::vector<double>> Z = A;
    for (auto& row : Z) {
        std::fill(row.begin(), row.end(), 0.0);
    }
    return Z;
}
std::vector<std::vector<std::vector<double>>> zeroLike3D(
    const std::vector<std::vector<std::vector<double>>>& A
) {
    std::vector<std::vector<std::vector<double>>> Z = A;
    for (auto& mat : Z) {
        for (auto& row : mat) {
            std::fill(row.begin(), row.end(), 0.0);
        }
    }
    return Z;
}
// ---------- add helpers ----------
void add1D(
    std::vector<double>& A,
    const std::vector<double>& B
) {
    if (A.size() != B.size()) {
        throw std::runtime_error("add1D: gradient dimensions do not match.");
    }

    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] += B[i];
    }
}
void add2D(
    std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B
) {
    if (!sameShape2D(A, B)) {
        throw std::runtime_error("add2D: gradient dimensions do not match.");
    }

    for (std::size_t i = 0; i < A.size(); ++i) {
        for (std::size_t j = 0; j < A[i].size(); ++j) {
            A[i][j] += B[i][j];
        }
    }
}
void add3D(
    std::vector<std::vector<std::vector<double>>>& A,
    const std::vector<std::vector<std::vector<double>>>& B
) {
    if (!sameShape3D(A, B)) {
        throw std::runtime_error("add3D: gradient dimensions do not match.");
    }

    for (std::size_t i = 0; i < A.size(); ++i) {
        for (std::size_t j = 0; j < A[i].size(); ++j) {
            for (std::size_t k = 0; k < A[i][j].size(); ++k) {
                A[i][j][k] += B[i][j][k];
            }
        }
    }
}
// ---------- scale helpers ----------
void scale1D(
    std::vector<double>& A,
    double factor
) {
    for (double& x : A) {
        x *= factor;
    }
}
void scale2D(
    std::vector<std::vector<double>>& A,
    double factor
) {
    for (auto& row : A) {
        for (double& x : row) {
            x *= factor;
        }
    }
}
void scale3D(
    std::vector<std::vector<std::vector<double>>>& A,
    double factor
) {
    for (auto& mat : A) {
        for (auto& row : mat) {
            for (double& x : row) {
                x *= factor;
            }
        }
    }
}
} // namespace

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
    bool storeEnergyHistory,
    bool useInteraction,
    bool useRBM,
    const std::vector<std::vector<double>>& a,
    const std::vector<double>& b,
    const std::vector<std::vector<std::vector<double>>>& W,
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

    if (useInteraction && !useRBM) {
        if (D != 3) {
            std::cerr << "Error: interacting hard-sphere implementation is currently only available for D=3.\n";
            std::exit(1);
        }
        if (mode == Mode::Importance) {
            std::cerr << "Error: interacting hard-sphere case is not yet implemented for importance sampling.\n";
            std::exit(1);
        }
    }

    auto rng = std::make_unique<Random>(seed);

    // ---------------- Particles ----------------
    std::vector<std::unique_ptr<Particle>> particles;

    if (useInteraction && !useRBM) {
        particles = setupRandomUniformInitialStateNoOverlap(initRange, D, N, hardCoreA, *rng);
    } else {
        particles = setupRandomUniformInitialState(initRange, D, N, *rng);
    }

    // ---------------- Wave function and Hamiltonian ----------------
    std::unique_ptr<Hamiltonian> hamiltonian;
    std::unique_ptr<WaveFunction> waveFunction;

    if (useRBM) {
        waveFunction = std::make_unique<BoltzmannMachine>(a, b, W, gamma);
        if (useInteraction) {
            hamiltonian = std::make_unique<RBMInteractingTrap>(omega, gamma, 1.0);
        } else {
            hamiltonian = std::make_unique<EllipticTrap>(gamma);
        }
    } 
    else {
        if (useInteraction) {
            waveFunction = std::make_unique<CorrelatedGaussian>(alpha, beta, hardCoreA);
        } else {
            waveFunction = std::make_unique<AnisotropicGaussian>(alpha, beta);
        }
        hamiltonian = std::make_unique<EllipticTrap>(gamma);
    }

    // ---------------- Solver ----------------
    // Move rng only after all particle initialization is finished.
    std::unique_ptr<MonteCarlo> solver;

    if (mode == Mode::Importance) {
        solver = std::make_unique<ImportanceSampling>(std::move(rng), 0.5);
    } else {
        solver = std::make_unique<Metropolis>(std::move(rng));
    }

    auto system = std::make_unique<System>(
        std::move(hamiltonian),
        std::move(waveFunction),
        std::move(solver),
        std::move(particles)
    );

    system->runEquilibrationSteps(stepParam, nEquil);

    if (useRBM) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto sampler = system -> runRBMMetropolisSteps(
            stepParam,
            nMetropolis,
            static_cast<unsigned int>(b.size()),
            storeEnergyHistory
        );
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dt = t1 - t0;
        RunResult res;
        res.cpu_seconds = dt.count();
        res.energy = sampler->getEnergy();
        res.gradienta = sampler -> getGradientA();
        res.gradientb = sampler -> getGradientB();
        res.gradientw = sampler -> getGradientW();
        res.acceptance = sampler->getAcceptanceRatio();

        // Store the energy values
        if (storeEnergyHistory){
            res.energyHistory = sampler ->getEnergyHistory();
        }

        return res;
    }
    else {
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
    
        return res;
    }
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
    int nReplicas,
    bool storeEnergyHistory,
    bool useInteraction,
    bool useRBM,
    const std::vector<std::vector<double>>& a,
    const std::vector<double>& b,
    const std::vector<std::vector<std::vector<double>>>& W,
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
            storeEnergyHistory,
            useInteraction,
            useRBM,
            a,
            b,
            W,
            beta,
            gamma,
            hardCoreA
        );
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    out.wall_seconds = dt.count();

    // Average the replica results
    if (useRBM){
        // The RBM has three different gradient objects:
        //   gradienta : N x D
        //   gradientb : Nh
        //   gradientw : N x D x Nh
        // We initialize the mean arrays using the dimensions of the first replica.

        if (out.replicas.empty()) {
            throw std::runtime_error("runVMCReplicasParallel: no replicas available.");
        }
        out.mean.gradienta = zeroLike2D(out.replicas[0].gradienta);
        out.mean.gradientb = zeroLike1D(out.replicas[0].gradientb);
        out.mean.gradientw = zeroLike3D(out.replicas[0].gradientw);

        for (const auto& replica : out.replicas) {
            out.mean.energy += replica.energy;
            add2D(out.mean.gradienta, replica.gradienta);
            add1D(out.mean.gradientb, replica.gradientb);
            add3D(out.mean.gradientw, replica.gradientw);
            out.mean.acceptance += replica.acceptance;
            out.mean.cpu_seconds += replica.cpu_seconds;
        }

        const double invReplicas = 1.0 / static_cast<double>(nReplicas);

        out.mean.energy *= invReplicas;
        scale2D(out.mean.gradienta, invReplicas);
        scale1D(out.mean.gradientb, invReplicas);
        scale3D(out.mean.gradientw, invReplicas);
        out.mean.acceptance *= invReplicas;
        out.mean.cpu_seconds *= invReplicas;

    }
    else {
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
    }

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

std::vector<std::vector<double>> matr_mult(std::vector<std::vector<double>>& a, std::vector<std::vector<double>>& b) {
    std::vector<std::vector<double>> res;
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < b.size(); j++) {
            for (int k = 0; k < a[0].size(); k++)
                res[i][j] += a[i][k] * b[k][j];
        }
    }

    return res;
}