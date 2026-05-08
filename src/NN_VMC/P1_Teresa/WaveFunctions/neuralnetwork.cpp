#include <memory>
#include <cmath>
#include <cassert>
#include <vector>
#include <stdexcept>

#include "neuralnetwork.h"
#include "../particle.h"

NeuralNetwork::NeuralNetwork(int64_t Nin, int64_t Nhid) {
    m_W1 = register_parameter("W1", torch::randn({ Nin, Nhid }));
    m_b = register_parameter("b", torch::randn(Nhid));
    m_W2 = register_parameter("W2", torch::randn(Nhid));
}

torch::Tensor NeuralNetwork::log_forward(torch::Tensor input) {
    // input is a [ 1 x Nin ] matrix
    torch::Tensor hidden = torch::addmm(m_b, input, m_W1);

    hidden = torch::tanh(hidden);
    torch::Tensor u_out = torch::mv(hidden, m_W2);

    return u_out;
}

torch::Tensor NeuralNetwork::forward(torch::Tensor input) {
    return torch::exp(log_forward(input));
}