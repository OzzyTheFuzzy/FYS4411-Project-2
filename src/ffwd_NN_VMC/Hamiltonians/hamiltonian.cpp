#include <memory>
#include <cassert>
#include <iostream>

#include "hamiltonian.h"

void Hamiltonian::set_analytic_ifAvailable(bool val) {
    this->m_analytic_ifAvailable = val;
}