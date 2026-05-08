#include <memory>
#include <cmath>
#include <cassert>
#include <vector>
#include <stdexcept>

#include "nn_envelope.h"
#include "../particle.h"

NN_envelope::NN_envelope(int N, int D, int Nin, int Nhid)
    : m_N(N), m_D(D), m_Nin(Nin), m_net(Nin, Nhid) {
    m_numberOfParameters = (Nin + 2) * Nhid;
}

torch::Tensor NN_envelope::encode(std::vector<std::unique_ptr<class Particle>>& particles) {
    std::vector<double> xi;
    xi.reserve(m_N * m_D);
    for (auto& p : particles)
        for (double x : p->getPosition())
            xi.push_back(x);

    // unsqueeze makes the [ m_N * m_D ] vector a [ 1 x m_N * m_D ] matrix
    return torch::tensor(xi).unsqueeze(0);
}

double NN_envelope::evaluate(std::vector<std::unique_ptr<class Particle>>& particles) {
    torch::NoGradGuard no_grad;
    auto input = encode(particles);
    return m_net.forward(input).item<double>();
}


// ── ∇²ψ / ψ  via autograd ─────────────────────────────────────────────────
// Uses the identity:
//   ∇²ψ / ψ  =  ∇²(log ψ)  +  |∇(log ψ)|²
// Both terms are computed from u_out = log ψ.
double NN_envelope::computeDoubleDerivative(
    std::vector<std::unique_ptr<class Particle>>& particles) {
    // Build position tensor with gradient tracking
    auto pos = encode(particles).squeeze(0)   // shape [Nin]
        .to(torch::kDouble)
        .requires_grad_(true);

    // log ψ — scalar
    auto log_psi = m_net.log_forward(pos.unsqueeze(0)).squeeze(0);

    // First derivatives ∂(log ψ)/∂pos_i,  shape [Nin]
    auto grad1 = torch::autograd::grad(
        { log_psi }, { pos },
        /*grad_outputs=*/{ torch::ones_like(log_psi) },
        /*retain_graph=*/true,
        /*create_graph=*/true   // needed so we can differentiate again
    )[0];

    // Second derivatives: diagonal of Hessian, shape [Nin]
    // Loop over each coordinate and extract its own second derivative
    double laplacian_log_psi = 0.0;
    for (int i = 0; i < m_Nin; ++i) {
        auto grad2 = torch::autograd::grad(
            { grad1[i] }, { pos },
            /*grad_outputs=*/{},
            /*retain_graph=*/true,
            /*create_graph=*/false
        )[0];
        laplacian_log_psi += grad2[i].item<double>();
    }

    double sq_grad = grad1.pow(2).sum().item<double>();   // |∇ log ψ|²

    return laplacian_log_psi + sq_grad;   // = ∇²ψ / ψ
}

std::vector<double> NN_envelope::computeLogParDer(
    std::vector<std::unique_ptr<class Particle>>& particles) {
    // Clear any gradients from previous calls
    m_net.zero_grad();

    // Parameters registered via register_parameter already have requires_grad=true
    // Positions don't need requires_grad here — we differentiate w.r.t. weights
    auto input = encode(particles);          // [1 x Nin]
    auto log_psi = m_net.log_forward(input);        // [1] — u_out = log ψ

    // Populates .grad() on W1, b, W2
    log_psi.backward();

    // order: W1, b, W2
    std::vector<double> OW(m_numberOfParameters);
    int idx = 0;

    for (const auto& param : m_net.parameters()) {
        auto grad = param.grad().flatten();
        for (int i = 0; i < grad.numel(); ++i) {
            OW[idx++] = grad[i].item<double>();
        }
    }

    return OW;
}