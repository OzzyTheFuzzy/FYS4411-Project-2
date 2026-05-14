#pragma once
#include <memory>
#include <vector>
#include <torch/torch.h>

#include "wavefunction.h"

class NeuralNetwork : public torch::nn::Module {
public:
    NeuralNetwork(int64_t Nin, int64_t Nhid, double helpDecay);
    NeuralNetwork(int64_t Nin, int64_t Nhid, double helpDecay, const std::vector<double>& params);
    torch::Tensor forward(torch::Tensor input);
    torch::Tensor log_forward(torch::Tensor input);

    void setGrads(const std::vector<double>& grads);

    std::vector<double> getParams() const;

private:
    int64_t m_Nin, m_Nhid;
    torch::Tensor m_W1;   // Nin x Nhid matrix
    torch::Tensor m_b;   // Nhid vector
    
    torch::Tensor m_W2;   // Nhid vector to out

    double m_helpDecay;
};