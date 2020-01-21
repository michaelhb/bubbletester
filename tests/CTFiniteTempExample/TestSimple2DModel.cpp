#include <Eigen/Core>
#include <iostream>
#include <sys/time.h>
#include <stdlib.h>
#include <vector> 

#include "GenericBounceSolver.hpp"
#include "BP1Driver.hpp"
#include "SimpleBounceDriver.hpp"
#include "BouncePath.hpp"
#include "Simple2DModel.hpp"

#define vacua(x1, x2) (Eigen::VectorXd(2) << x1, x2).finished()

int n_spatial_dimensions = 3;

namespace BubbleTester {

struct TestPoint {
    double T;
    Eigen::VectorXd low_vevs;
    Eigen::VectorXd high_vevs;
    double CTAction; // CosmoTransitions result
};

void run_test(std::vector<TestPoint> tests, Simple2DModel model, 
    std::shared_ptr<GenericBounceSolver> solver, bool normalise, bool plot=false) {

    Eigen::VectorXd origin = Eigen::VectorXd::Zero(2);
    double rescale = 1.0;
    
    for (auto& test : tests) {
        FiniteTempPotential potential = FiniteTempPotential(model, test.T);

        Eigen::VectorXd true_vacuum = test.low_vevs;
        Eigen::VectorXd false_vacuum = test.high_vevs;

        if (normalise) {
            rescale = GenericPotential::normalise(potential, n_spatial_dimensions, test.low_vevs, test.high_vevs);
            std::cout << "rescale factor: " << rescale << std::endl;
            true_vacuum = (Eigen::VectorXd(2) << 1., 0.).finished();
            false_vacuum = origin;
        }
        
        // Solve the bounce
        BouncePath path;
        bool success = false;

        try {
            std::cout << "T = " << test.T << std::endl;
            path = solver->solve(true_vacuum, false_vacuum, potential);
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
            
            double margin;

            if (normalise) {
                margin = 0.5;
            }
            else {
                margin = (test.low_vevs - test.high_vevs).norm() / 10;
            }
            
            if (success) {
                title << "T = ";
                title << test.T;
                potential.plot_2d(title.str(), 200, true_vacuum, origin, margin, path);
            }
            else {
                title << "T = ";
                title << test.T;
                potential.plot_2d(title.str(), 200, true_vacuum, origin, 0.5);
            }
        }    
    }
}

};

int main() {
    using namespace BubbleTester;

    std::vector<TestPoint> tests;

    tests.push_back(
        TestPoint{78.0, vacua(289.13152298, 389.53915109), vacua(233.91355645, -118.2634663), 1042.9636324459277});
    tests.push_back(
        TestPoint{80.0, vacua(288.33310414, 387.40609918), vacua(232.8789432, -128.14426035), 4494.24682584152});
    tests.push_back(
        TestPoint{82.0, vacua(287.46352188, 385.07185158), vacua(232.01916761, -133.06697241), 7784.151053576536});
    tests.push_back(
        TestPoint{84.0, vacua(286.51619595, 382.51685993), vacua(231.19950717, -136.47846514), 11342.167512200098});
    tests.push_back(
        TestPoint{86.0, vacua(285.48755338, 379.72446235), vacua(230.39178861, -139.07629584), 15332.704201800192});
    tests.push_back(
        TestPoint{88.0, vacua(284.3691753, 376.66976324), vacua(229.58404847, -141.14357714), 20120.14839389324});
    tests.push_back(
        TestPoint{90.0, vacua(283.1544923, 373.32764204), vacua(228.76961321, -142.82630305), 25900.088264367914});
    tests.push_back(
        TestPoint{92.0, vacua(281.83496523, 369.66772314), vacua(227.946157, -144.36962661), 33046.39135123631});
    tests.push_back(
        TestPoint{94.0, vacua(280.40042234, 365.65210141), vacua(227.11058263, -145.69719473), 42327.127602430905});
    tests.push_back(
        TestPoint{96.0, vacua(278.83936508, 361.23669465), vacua(226.25825717, -146.72933974), 55052.35221634127});
    tests.push_back(
        TestPoint{98.0, vacua(277.13734813, 356.36650468), vacua(225.38835079, -147.58226411), 73551.10574804063});
    tests.push_back(
        TestPoint{100.0, vacua(275.277131, 350.97100965), vacua(224.4992334, -148.28019508), 102660.40028831027});
    tests.push_back(
        TestPoint{102.0, vacua(273.23607878, 344.95992658), vacua(223.58953324, -148.84547219), 153512.8446612609});
    tests.push_back(
        TestPoint{104.0, vacua(270.98546769, 338.21141955), vacua(222.65794425, -149.29368858), 261867.60491844645});
    tests.push_back(
        TestPoint{106.0, vacua(268.48710682, 330.55764111), vacua(221.70370405, -149.6397163), 583694.5594380731});

    // set up the model
    double m1 = 120.;
    double m2 = 50.;
    double mu = 25.;
    double Y1 = .1;
    double Y2 = .15;
    int n = 30.;
    double renorm_scale = 246.;

    Simple2DModel model = Simple2DModel(m1, m2, mu, Y1, Y2, n, renorm_scale);

    std::cout << "Testing BubbleProfiler V1:" << std::endl;
    std::shared_ptr<GenericBounceSolver> bp_solver = std::make_shared<BP1BounceSolver>(n_spatial_dimensions);
    bp_solver->set_verbose(true);
    run_test(tests, model, bp_solver, true, true);
    

    // std::cout << "Testing SimpleBounce:" << std::endl;
    // std::shared_ptr<GenericBounceSolver> sb_solver = std::make_shared<SimpleBounceSolver>(10., 100., n_spatial_dimensions);
    // sb_solver->set_verbose(true);
    // // run_test(tests, model, sb_solver, true);
    // run_test_normalised(tests, model, sb_solver, true);
}

