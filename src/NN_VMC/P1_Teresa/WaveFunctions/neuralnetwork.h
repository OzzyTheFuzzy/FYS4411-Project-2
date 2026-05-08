#pragma once
#include <memory>
#include <vector>
#include <torch/torch.h>

#include "wavefunction.h"

class NeuralNetwork : public torch::nn::Module {
public:
    NeuralNetwork(int64_t N, int64_t M);
    torch::Tensor forward(torch::Tensor input);
    torch::Tensor log_forward(torch::Tensor input);

private:
    torch::Tensor m_W1;   // Nin x Nhid matrix
    torch::Tensor m_b;   // Nhid vector
    
    torch::Tensor m_W2;   // Nhid vector to out
};