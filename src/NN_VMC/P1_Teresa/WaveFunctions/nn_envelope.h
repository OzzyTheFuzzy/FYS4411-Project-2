#pragma once
#include <memory>
#include <vector>

#include "wavefunction.h"
#include "neuralnetwork.h"

class NN_envelope : public WaveFunction {
public:
    NN_envelope(int N, int D, int Nin, int Nhid);

    double evaluate(std::vector<std::unique_ptr<class Particle>>& particles) override;
    torch::Tensor encode(std::vector<std::unique_ptr<class Particle>>& particles);
    // std::vector<double> QFac(std::vector<std::unique_ptr<class Particle>>& particles);
    double computeDoubleDerivative(std::vector<std::unique_ptr<class Particle>>& particles) override;
    // std::vector<double> computeQuantumForce(std::vector<std::unique_ptr<class Particle>>& particles,unsigned int particleIndex) override;
    
    std::vector<double> computeLogParDer(std::vector<std::unique_ptr<class Particle>>& particles);

    NeuralNetwork& net() { return m_net; }

private:
    int m_N;
    int m_D;
    int m_Nin;
    NeuralNetwork m_net;
};