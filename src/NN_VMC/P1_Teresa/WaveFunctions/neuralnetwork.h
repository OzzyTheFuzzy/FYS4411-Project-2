#pragma once
#include <memory>
#include <vector>
#include <torch/torch.h>

#include "wavefunction.h"

class NeuralNetwork : public torch::nn::Module {
public:
    NeuralNetwork(int64_t N, int64_t M);
    torch::Tensor forward(torch::Tensor input);

private:
    torch::Tensor W1;   // Nin x Nhid matrix
    torch::Tensor b;   // Nhid vector
    
    torch::Tensor W2;   // Nhid vector to out
};