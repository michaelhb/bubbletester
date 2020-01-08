#include <Eigen/Core>
#include <iostream>

#include "QuarticPotential.hpp"
#include "GenericBounceSolver.hpp"
#include "BP1Driver.hpp"
#include "SimpleBounceDriver.hpp"
#include "BouncePath.hpp"

// TODO: Add more solvers, clean up console output. 

namespace BubbleTester { 

void thin_wall_test(std::shared_ptr<GenericBounceSolver> solver) {
    // Find the smallest value of alpha for which the solver works,
    // using a bisection algorithm. 

    Eigen::VectorXd false_vacuum(1), true_vacuum(1);
    false_vacuum << 0.; 
    true_vacuum << 1.;

    double lower = 0.5;
    double upper = 0.75;
    double bisect;
    double tol = 0.000001;

    BouncePath path;

    while(upper - lower > tol) {
        bisect = (upper + lower) / 2.0;

        std::cout << "Trying alpha = " << bisect << std::endl;

        QuarticPotential potential(bisect);

        try {
            path = solver->solve(true_vacuum, false_vacuum, potential);
            upper = bisect;
        }
        catch (const std::exception& e) {
            lower = bisect;
        }
    }

    std::cout << "Final alpha: " << upper << std::endl;
}

};

int main() {
    using namespace BubbleTester;

    // std::cout << "Testing BubbleProfiler V1:" << std::endl;
    // std::shared_ptr<GenericBounceSolver> bp_solver = std::make_shared<BP1BounceSolver>();
    // thin_wall_test(bp_solver);

    std::cout << "Testing SimpleBounce:" << std::endl;
    std::shared_ptr<GenericBounceSolver> sb_solver = std::make_shared<SimpleBounceSolver>(50, 100);
    thin_wall_test(sb_solver);
}

