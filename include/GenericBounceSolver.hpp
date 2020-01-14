#ifndef BUBBLETESTER_GENERICBOUNCESOLVER_HPP_INCLUDED
#define BUBBLETESTER_GENERICBOUNCESOLVER_HPP_INCLUDED

#include <Eigen/Core>

#include "BouncePath.hpp"
#include "GenericPotential.hpp"

namespace BubbleTester {

class GenericBounceSolver {
public:
    virtual BouncePath solve(
        const Eigen::VectorXd& true_vacuum,
        const Eigen::VectorXd& false_vacuum,
        const GenericPotential& potential) const = 0;

    virtual void set_verbose(bool verbose) = 0;
};

};

#endif