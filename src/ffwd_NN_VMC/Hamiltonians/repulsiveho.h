#pragma once
#include <memory>
#include <vector>

#include "hamiltonian.h"

/**
 * @brief Represents an interacting Bose gas in an elliptical harmonic trap.
 * * This Hamiltonian introduces a hard-core repulsive potential between particles
 * (the Jastrow factor physics) and allows for asymmetrical trapping frequencies
 * (omega_z != omega_perp)
 */
class RepulsiveHO : public Hamiltonian {
public:
    /**
     * @brief Constructs a spherical trap with default hard-core repulsion.
     */
    RepulsiveHO(double omega);

    /**
     * @brief Constructs an elliptical trap with default hard-core repulsion.
     */
    RepulsiveHO(double omega, double omega_z);

    /**
     * @brief Constructs an elliptical trap specifying the hard-core radius.
     * @param omega Trap frequency in the xy-plane.
     * @param omega_z Trap frequency along the z-axis.
     * @param repulsive_a_factor The interaction diameter 'a' in standard units.
     */
    RepulsiveHO(double omega, double omega_z, double repulsive_a_factor);

    /**
     * @brief Constructs an elliptical trap specifying the hard-core radius.
     * @param omega Trap frequency in the xy-plane.
     * @param omega_z Trap frequency along the z-axis.
     * @param repulsive_a_factor The interaction diameter 'a' in standard units.
     * @param hardcore_strength Maximum interaction strength.
     */
    RepulsiveHO(double omega, double omega_z, double repulsive_a_factor,
        double hardcore_strength);

    double computeLocalEnergy(
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles
    ) override;

    double computeLocalEnergy(
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles,
        class WaveFunctionCache& cache
    ) override;

    double getRepulsiveFactor() override { return m_rep_a; }

    double has_hardcore() override {
        return true;
    }

    double get_interaction_strength() override { return m_strength; }

    void set_interaction_strength(double strength) override {
        m_strength = strength;
    }

private:
    double m_omega;
    double m_omega_z;
    double m_rep_a;     ///< Scaled hard-core diameter
    double m_strength = std::numeric_limits<double>::infinity();
};

