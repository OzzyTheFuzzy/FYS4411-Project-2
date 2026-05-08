#include <memory>
#include <cmath>
#include <cassert>
#include <vector>
#include <stdexcept>

#include "neuralnetwork.h"
#include "../particle.h"

NeuralNetwork::NeuralNetwork(int64_t Nin, int64_t Nhid) {
    W1 = register_parameter("W1", torch::randn({ Nin, Nhid }));
    b = register_parameter("b", torch::randn(Nhid));
    W2 = register_parameter("W2", torch::randn(Nhid));
}

torch::Tensor NeuralNetwork::forward(torch::Tensor input) {
    torch::Tensor hidden = torch::addmm(b, input, W1);

    hidden = torch::tanh(hidden);
    torch::Tensor u_out = torch::mv(hidden, W2);

    return torch::exp(u_out);
}