#include <Eigen/Core>
#include <iostream>
#include <sys/time.h>

#include "QuarticPotential.hpp"
#include "GenericBounceSolver.hpp"
#include "BP1Driver.hpp"
#include "SimpleBounceDriver.hpp"
#include "BouncePath.hpp"

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

        std::cout << "Trying alpha = " << bisect << " ";

        QuarticPotential potential(bisect);

        try {
            struct timeval time1;
	        struct timeval time2;

            gettimeofday(&time1, NULL);
            path = solver->solve(true_vacuum, false_vacuum, potential);
            gettimeofday(&time2, NULL);

            upper = bisect;
            float elapsed = time2.tv_sec - time1.tv_sec 
                +  (float)(time2.tv_usec - time1.tv_usec) / 1000000;

            std::cout << "-> action: " << path.get_action() << " in " << elapsed << "s" << std::endl;
        }
        catch (const std::exception& e) {
            lower = bisect;
            std::cout << "failed" << std::endl;
        }
    }

    std::cout << "Final alpha: " << upper << std::endl;
}

};

int main() {
    using namespace BubbleTester;

    std::cout << "Testing BubbleProfiler V1:" << std::endl;
    std::shared_ptr<GenericBounceSolver> bp_solver = std::make_shared<BP1BounceSolver>();
    thin_wall_test(bp_solver);

    std::cout << "Testing SimpleBounce:" << std::endl;
    std::shared_ptr<GenericBounceSolver> sb_solver = std::make_shared<SimpleBounceSolver>(1., 100., false);
    thin_wall_test(sb_solver);
}

