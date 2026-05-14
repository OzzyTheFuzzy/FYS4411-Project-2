#include <iostream>
#include <iomanip>
#include <nlopt.hpp>

#include "VMCOptimizer.h"
#include "system.h"
#include "Samplers/energysampler.h"
#include "WaveFunctions/ellipticgaussian.h"
#include "InitialStates/initialstate.h"
#include "Math/random.h"
#include "harmonicoscillator.h"
#include "Solvers/metropolishastings.h"

VMCOptimizer::VMCOptimizer(
    MCEngine& engine,
    unsigned int numberOfMetropolisSteps,
    double BFGS_tol,
    std::ofstream* logfile,
    std::ofstream* outfile,
    std::ofstream* paramsfile
) : m_engine(engine)
, m_numberOfMetropolisSteps(numberOfMetropolisSteps)
, m_BFGS_tol(BFGS_tol)
, m_logfile(logfile)
, m_outfile(outfile) 
, m_paramsfile(paramsfile) {}

double VMCOptimizer::computeMC(const std::vector<double>& params, std::vector<double>& grad) {
    std::cout << "\rComputing MC #" << ++m_mcCount << std::flush;

    auto sampler = m_engine.run(params, m_numberOfMetropolisSteps);

    if (!grad.empty()) {
        for (unsigned int i = 0; i < params.size(); i++) {
            grad[i] = 2 * sampler->getCovariance(i);
        }
    }

    if (m_logfile) sampler->logOutput(params, *m_logfile);

    return sampler->getEnergy();
}

std::vector<double> VMCOptimizer::optimize(std::vector<double> params) {
    // print log header
    if (m_logfile) {
        *m_logfile << "#";
        const unsigned int width = 20;
        for (unsigned int i = 0; i < params.size(); i++) {
            std::string temp = "p[" + std::to_string(i) + "],";
            *m_logfile << std::setw(width - (i == 0)) << temp;
        }
        *m_logfile << std::setw(width) << "energy," << std::setw(width) << "variance,"
            << std::setw(width) << "error," << std::setw(width) << "elapsed time,"
            << std::setw(width) << "acceptance ratio " << std::endl;
    }

    nlopt::opt lib_optimizer(nlopt::LD_LBFGS, params.size());
    {
        auto tempWaveFunction = m_engine.makeWaveFunction(params);
        auto lb = tempWaveFunction->lowerBounds();
        auto ub = tempWaveFunction->upperBounds();
        if (!lb.empty()) lib_optimizer.set_lower_bounds(lb);
        if (!ub.empty()) lib_optimizer.set_upper_bounds(ub);
    }
    lib_optimizer.set_min_objective(nloptObjective, this);
    lib_optimizer.set_xtol_rel(m_BFGS_tol);
    lib_optimizer.set_maxeval(400);
    lib_optimizer.set_maxtime(10800.0);
    m_mcCount = 0;

    double minEnergy;
    try {
        lib_optimizer.optimize(params, minEnergy);
    }
    catch (const std::runtime_error& e) {
        std::cout << "\nNLopt failed: " << e.what() << std::endl;
        std::cout << "Last energy: " << minEnergy << std::endl;
        std::cout << "Last params: ";
        for (auto p : params) std::cout << p << " ";
        if (m_outfile) {
            *m_outfile << "# NLopt failed: " << e.what() << std::endl;
            *m_outfile << "# Last energy: " << minEnergy << std::endl;
        }
    }
    std::cout << std::endl;

    if (m_outfile) {
        *m_outfile << "# Optimal energy: " << minEnergy << std::endl;
        *m_outfile << "# Optimal parameters: " << std::setprecision(8);
        for (auto p : params) *m_outfile << p << ", \t";
        *m_outfile << std::endl;
    }
    if (m_paramsfile) {
        for (auto p : params) *m_paramsfile << p << std::endl;
    }

    return params;
}