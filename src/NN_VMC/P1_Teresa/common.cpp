#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#include "common.h"
#include "WaveFunctions/neuralnetwork.h"

using namespace CommonUtils;

std::vector<double> readVector(const std::string & filename) {
    std::ifstream file(filename);
    std::vector<double> vec;
    double temp;
    while (file >> temp)
        vec.push_back(temp);
    file.close();
    return vec;
}

std::vector<std::vector<double>> readMatrix(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> mat;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;  // skip empty lines and comments
        std::vector<double> row;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try { row.push_back(std::stod(token)); }
            catch (...) { continue; }  // skip malformed tokens
        }
        if (!row.empty()) mat.push_back(row);
    }
    file.close();
    return mat;
}

std::pair<double, double> mean_err(std::vector<double>& vec) {
    double sum = 0;
    for (double val : vec) {
        sum += val;
    }
    double mean = sum / (double)vec.size();
    sum = 0;
    for (double val : vec) {
        sum += sq(val - mean);
    }
    double var = sum / (double)(vec.size() - 1);
    double err = sqrt(var) / sqrt((double)vec.size());
    return std::make_pair(mean, err);
}

void setNetworkGrads(NeuralNetwork& net, const std::vector<double>& grads) {
    int idx = 0;
    for (auto& param : net.parameters()) {
        int n = param.numel();
        auto grad_tensor = torch::zeros_like(param);
        auto acc = grad_tensor.accessor<float, 1>();  // adjust rank if W1 is 2D
        
        // Use flatten + copy to handle any parameter shape
        auto flat = torch::tensor(
            std::vector<float>(grads.begin() + idx, grads.begin() + idx + n)
        ).reshape_as(param);

        param.mutable_grad() = flat;
        idx += n;
    }
}