#include <Eigen/Core>
#include <iostream>
#include <sys/time.h>

#include "GenericBounceSolver.hpp"
#include "BP1Driver.hpp"
#include "SimpleBounceDriver.hpp"
#include "BouncePath.hpp"
#include "CTAnalyticPotential.hpp"

int n_spatial_dimensions = 3;

namespace BubbleTester {

void run_test(std::shared_ptr<GenericBounceSolver> solver, double delta, bool normalise, bool plot=false) {
    CTAnalyticPotential potential = CTAnalyticPotential(delta);

    Eigen::VectorXd origin = Eigen::VectorXd::Zero(2);

    // Find the location of the true vacuum
    Eigen::VectorXd ub(2);
    Eigen::VectorXd lb(2);
    Eigen::VectorXd start(2);
    ub << 2., 2.;
    lb << .5, .5;
    start << 1., 1.;

    Eigen::VectorXd true_vacuum = potential.minimise(start, lb, ub);
    std::cout << "True vacuum: " << true_vacuum << std::endl;

    double rescale = 1.0;

    // Normalise if required
    if (normalise) {
        rescale = GenericPotential::normalise(potential, n_spatial_dimensions, true_vacuum, origin);
        true_vacuum = (Eigen::VectorXd(2) << 1., 0.).finished();
        std::cout << "Normalised potential, rescale factor = " << rescale << std::endl;
    }

    // Solve the bounce
    std::cout << "Testing " << solver->name() << ": ";
    BouncePath path;
    bool success = false;

    try {
        path = solver->solve(true_vacuum, origin, potential);
        success = true;
        std::cout << "action = " << path.get_action() << std::endl;
        std::cout << "rescaled = " << path.get_action()*rescale << std::endl;
    }
    catch (...) {
        std::cout << "failed" << std::endl;
    }

    if (plot) {
        std::cout << "Plotting result..." << std::endl;
        std::ostringstream title;
        
        if (success) {
            title << "Delta = " << delta;
            potential.plot_2d(title.str(), 200, true_vacuum, origin, 0.5, {path});
        }
        else {
            title << "Delta = " << delta << " (no solution found)";
            potential.plot_2d(title.str(), 200, true_vacuum, origin, 0.5);
        }
    }

}

};
int main() {
    using namespace BubbleTester;

    std::shared_ptr<GenericBounceSolver> bp_solver = std::make_shared<BP1BounceSolver>(n_spatial_dimensions);
    std::shared_ptr<GenericBounceSolver> sb_solver = std::make_shared<SimpleBounceSolver>(1., 100., n_spatial_dimensions);

    bp_solver->set_verbose(true);
    sb_solver->set_verbose(true);

    double delta = 0.4;

    // run_test(bp_solver, delta, false, true);
    // run_test(bp_solver, delta, true, true);
    run_test(sb_solver, delta, false, true);
    // run_test(sb_solver, delta, true, true);
    
}