#pragma once

#include <cmath>
#include <vector>
#include <stdexcept>

// -----------------------------------------------------------------------------
// Adam optimizer for the RBM variational parameters a, b, W.
//
// The RBM-VMC sampler returns the energy gradient arrays
//
//   gradA[p][d]      = d<E>/d a[p][d]
//   gradB[h]         = d<E>/d b[h]
//   gradW[p][d][h]   = d<E>/d W[p][d][h]
//
// Adam minimizes the energy by applying
//
//   theta <- theta - lr_t * m / (sqrt(v) + eps)
//
// separately to a, b, and W.
// -----------------------------------------------------------------------------
class RBMAdamOptimizer {
public:
    RBMAdamOptimizer(
        const std::vector<std::vector<double>>& a,
        const std::vector<double>& b,
        const std::vector<std::vector<std::vector<double>>>& W,
        double learningRate = 0.01,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double epsilon = 1e-8
    ) : m_lr(learningRate), m_beta1(beta1), m_beta2(beta2), m_eps(epsilon), m_t(0)
    {
        if (learningRate <= 0.0) {
            throw std::invalid_argument("RBMAdamOptimizer: learningRate must be positive.");
        }
        if (beta1 < 0.0 || beta1 >= 1.0 || beta2 < 0.0 || beta2 >= 1.0) {
            throw std::invalid_argument("RBMAdamOptimizer: beta1 and beta2 must be in [0,1).");
        }
        if (epsilon <= 0.0) {
            throw std::invalid_argument("RBMAdamOptimizer: epsilon must be positive.");
        }

        m_mA = zerosLike(a);
        m_vA = zerosLike(a);
        m_mB = std::vector<double>(b.size(), 0.0);
        m_vB = std::vector<double>(b.size(), 0.0);
        m_mW = zerosLike(W);
        m_vW = zerosLike(W);
    }

    void update(
        std::vector<std::vector<double>>& a,
        std::vector<double>& b,
        std::vector<std::vector<std::vector<double>>>& W,
        const std::vector<std::vector<double>>& gradA,
        const std::vector<double>& gradB,
        const std::vector<std::vector<std::vector<double>>>& gradW
    ) {
        m_t++;
        update2D(a, gradA, m_mA, m_vA);
        update1D(b, gradB, m_mB, m_vB);
        update3D(W, gradW, m_mW, m_vW);
    }

private:
    double m_lr;
    double m_beta1;
    double m_beta2;
    double m_eps;
    int m_t;

    std::vector<std::vector<double>> m_mA;
    std::vector<std::vector<double>> m_vA;
    std::vector<double> m_mB;
    std::vector<double> m_vB;
    std::vector<std::vector<std::vector<double>>> m_mW;
    std::vector<std::vector<std::vector<double>>> m_vW;

    static std::vector<std::vector<double>> zerosLike(
        const std::vector<std::vector<double>>& x
    ) {
        std::vector<std::vector<double>> z = x;
        for (auto& row : z) {
            for (auto& val : row) val = 0.0;
        }
        return z;
    }

    static std::vector<std::vector<std::vector<double>>> zerosLike(
        const std::vector<std::vector<std::vector<double>>>& x
    ) {
        std::vector<std::vector<std::vector<double>>> z = x;
        for (auto& mat : z) {
            for (auto& row : mat) {
                for (auto& val : row) val = 0.0;
            }
        }
        return z;
    }

    double correctedStepSize() const {
        // Bias-corrected Adam learning rate.
        return m_lr * std::sqrt(1.0 - std::pow(m_beta2, m_t)) /
                      (1.0 - std::pow(m_beta1, m_t));
    }

    void updateScalar(double& param, double grad, double& m, double& v, double lr_t) {
        m = m_beta1 * m + (1.0 - m_beta1) * grad;
        v = m_beta2 * v + (1.0 - m_beta2) * grad * grad;
        param -= lr_t * m / (std::sqrt(v) + m_eps);
    }

    void update1D(
        std::vector<double>& params,
        const std::vector<double>& grads,
        std::vector<double>& m,
        std::vector<double>& v
    ) {
        if (params.size() != grads.size()) {
            throw std::runtime_error("RBMAdamOptimizer: 1D gradient size mismatch.");
        }
        const double lr_t = correctedStepSize();
        for (std::size_t i = 0; i < params.size(); ++i) {
            updateScalar(params[i], grads[i], m[i], v[i], lr_t);
        }
    }

    void update2D(
        std::vector<std::vector<double>>& params,
        const std::vector<std::vector<double>>& grads,
        std::vector<std::vector<double>>& m,
        std::vector<std::vector<double>>& v
    ) {
        if (params.size() != grads.size()) {
            throw std::runtime_error("RBMAdamOptimizer: 2D gradient size mismatch.");
        }
        const double lr_t = correctedStepSize();
        for (std::size_t i = 0; i < params.size(); ++i) {
            if (params[i].size() != grads[i].size()) {
                throw std::runtime_error("RBMAdamOptimizer: 2D gradient row-size mismatch.");
            }
            for (std::size_t j = 0; j < params[i].size(); ++j) {
                updateScalar(params[i][j], grads[i][j], m[i][j], v[i][j], lr_t);
            }
        }
    }

    void update3D(
        std::vector<std::vector<std::vector<double>>>& params,
        const std::vector<std::vector<std::vector<double>>>& grads,
        std::vector<std::vector<std::vector<double>>>& m,
        std::vector<std::vector<std::vector<double>>>& v
    ) {
        if (params.size() != grads.size()) {
            throw std::runtime_error("RBMAdamOptimizer: 3D gradient size mismatch.");
        }
        const double lr_t = correctedStepSize();
        for (std::size_t i = 0; i < params.size(); ++i) {
            if (params[i].size() != grads[i].size()) {
                throw std::runtime_error("RBMAdamOptimizer: 3D gradient row-size mismatch.");
            }
            for (std::size_t j = 0; j < params[i].size(); ++j) {
                if (params[i][j].size() != grads[i][j].size()) {
                    throw std::runtime_error("RBMAdamOptimizer: 3D gradient depth-size mismatch.");
                }
                for (std::size_t k = 0; k < params[i][j].size(); ++k) {
                    updateScalar(params[i][j][k], grads[i][j][k], m[i][j][k], v[i][j][k], lr_t);
                }
            }
        }
    }
};
