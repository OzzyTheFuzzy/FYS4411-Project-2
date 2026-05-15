#pragma once
#include <memory>
#include <vector>
#include <limits>

#include "hamiltonian.h"

/**
 * @brief Represents an interacting Bose gas in an elliptical harmonic trap.
 * * This Hamiltonian introduces a hard-core repulsive potential between particles
 * (the Jastrow factor physics) and allows for asymmetrical trapping frequencies
 * (omega_z != omega_perp)
 */
class CoulombHO : public Hamiltonian {
public:
    /**
     * @brief Constructs a spherical trap with default hard-core repulsion.
     */
    CoulombHO(double omega);

    /**
     * @brief Constructs an elliptical trap with default hard-core repulsion.
     */
    CoulombHO(double omega, double omega_z);

    /**
     * @brief Constructs an elliptical trap specifying the hard-core radius.
     * @param omega Trap frequency in the xy-plane.
     * @param omega_z Trap frequency along the z-axis.
     * @param maxStrength Coulomb interaction maxStrength.
     */
    CoulombHO(double omega, double omega_z, double maxStrength);

    /**
     * @brief Constructs an elliptical trap specifying the hard-core radius.
     * @param omega Trap frequency in the xy-plane.
     * @param omega_z Trap frequency along the z-axis.
     * @param maxStrength Coulomb interaction maxStrength.
     * @param percStrength Percentage of interaction strength.
     */
    CoulombHO(double omega, double omega_z, double maxStrength,
        double percStrength);

    double computeLocalEnergy(
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles
    ) override;

    double computeLocalEnergy(
        class WaveFunction& waveFunction,
        std::vector<std::unique_ptr<class Particle>>& particles,
        class WaveFunctionCache& cache
    ) override;

    double get_interaction_strength() override;

    void set_percStrength(double percStrength) override;

private:
    double m_omega;
    double m_omega_z;
    double m_maxStrength = 1;
    double m_percStrength = 1;

    const double c_eps = 1e-12; // against numerical errors
};

