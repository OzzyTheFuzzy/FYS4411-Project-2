#pragma once

#include <vector>

/**
 * @brief Namespace containing general mathematical utility functions.
 */
namespace CommonUtils {
    /**
     * @brief Efficiently calculates the square of a value.
     * @tparam T Numeric type (e.g., double, int).
     * @param x The value to square.
     * @return The squared value ($x^2$).
     */
    template <typename T>
    constexpr T sq(T x) {
        return x * x;
    }
}

/**
 * @brief Reads a 1D vector of data from a text file.
 * @param filename The path to the file to read.
 * @return A vector of doubles containing the read values.
 */
std::vector<double> readVector(const std::string& filename);

/**
 * @brief Reads a 2D matrix of data from a structured file (e.g., CSV).
 * @param filename The path to the file to read.
 * @return A vector of vectors representing the matrix.
 */
std::vector<std::vector<double>> readMatrix(const std::string& filename);

/**
 * @brief Calculates the mean and standard error of a data sample.
 * @param vec Reference to the raw data vector to analyze.
 * @return A pair where `first` is the mean and `second` is the standard error.
 */
std::pair<double, double> mean_err(std::vector<double>& vec);