#include <casadi/casadi.hpp>
#include <Eigen/Core>
#include <chrono>
#include "BouncePath.hpp"
#include "CasadiPotential.hpp"
#include "CasadiMaupertuisDriver.hpp"
#include "CasadiCollocationDriver.hpp"
#include "CasadiCollocationDriver2.hpp"
#include "BP1Driver.hpp"
#include "SimpleBounceDriver.hpp"

#define sqr(x) (x)*(x)

casadi::Function get_potential(double delta) {
    using namespace casadi;
    SX phi_1 = SX::sym("phi_1");
    SX phi_2 = SX::sym("phi_2");

    SX phi = SX::vertcat(SXVector({phi_1, phi_2}));
    SX V = (sqr(phi_1) + sqr(phi_2))*(1.8*sqr(phi_1 - 1) + 0.2*sqr(phi_2 - 1) - delta);
    return Function("fV", {phi}, {V}, {"phi"}, {"V"});
}

int main() {
    // using namespace casadi;
    using namespace BubbleTester;
    using namespace std::chrono;
    double delta = 0.3;
    casadi::Function fPotential = get_potential(delta);
    CasadiPotential potential = CasadiPotential(fPotential, 2);
    
    Eigen::VectorXd origin = Eigen::VectorXd::Zero(2);

    // Find the location of the true vacuum
    Eigen::VectorXd ub(2);
    Eigen::VectorXd lb(2);
    Eigen::VectorXd start(2);
    ub << 2., 2.;
    lb << .5, .5;
    start << 1., 1.;
    
    Eigen::VectorXd true_vacuum = potential.minimise(start, lb, ub);
    std::cout << "True vacuum:" << std::endl << true_vacuum << std::endl;

    // Thin wall solver
    // std::shared_ptr<GenericBounceSolver> mp_solver = std::make_shared<CasadiMaupertuisSolver>(2);	
    // auto t1 = high_resolution_clock::now();
    // BouncePath mp_path = mp_solver->solve(true_vacuum, origin, potential);
    // auto t2 = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(t2 - t1);
    // std::cout << "GenericPotential::solve: " << duration.count()*1e-6 << " sec" << std::endl;
    // potential.plot_2d("CasADi thin wall limit", 200, true_vacuum, origin, 0.1, {mp_path});

    // Collocation Solver
    // std::shared_ptr<GenericBounceSolver> c_solver = std::make_shared<CasadiCollocationSolver>(3, 50);
    // c_solver->set_verbose(true);
    // BouncePath c_path = c_solver->solve(true_vacuum, origin, potential);
    // std::cout << "Action = " << c_path.get_action() << std::endl;
    // // std::cout << "Radii: " << c_path.get_radii() << std::endl;
    // // std::cout << "Profiles: " << c_path.get_profiles() << std::endl;
    // c_path.plot_profiles(20., "Collocation Solver (unscaled)");

    // Collocation Solver 2
    std::shared_ptr<GenericBounceSolver> c2_solver = std::make_shared<CasadiCollocationSolver2>(2, 3, 100);
    c2_solver->set_verbose(true);
    BouncePath c2_path = c2_solver->solve(true_vacuum, origin, potential);
    std::cout << "Action = " << c2_path.get_action() << std::endl;
    std::ostringstream title;
    title << "Collocation solver: delta = " << delta << ", action = " << c2_path.get_action();
    c2_path.plot_profiles(10., title.str());

    // BubbleProfiler
    // std::shared_ptr<GenericBounceSolver> bp_solver = std::make_shared<BP1BounceSolver>(2);
    // bp_solver->set_verbose(true);
    // BouncePath bp_path = bp_solver->solve(true_vacuum, origin, potential);
    // potential.plot_2d("BubbleProfiler solution", 200, true_vacuum, origin, 0.5, {bp_path});
    // std::cout << "Action = " << bp_path.get_action() << std::endl;
    // std::cout << "Radii:" << std::endl << bp_path.get_radii() << std::endl;
    // std::cout << "Profiles:" << std::endl << bp_path.get_profiles() << std::endl;

    // SimpleBounce
    std::shared_ptr<GenericBounceSolver> sb_solver = std::make_shared<SimpleBounceSolver>(150., 100., 3);
    sb_solver->set_verbose(true);
    BouncePath sb_path = sb_solver->solve(true_vacuum, origin, potential);
    // potential.plot_2d("SB Solution", 200, true_vacuum, origin, 0.1, {sb_path});
    // std::cout << "Action = " << sb_path.get_action() << std::endl;
    // // std::cout << "Radii:" << std::endl << sb_path.get_radii() << std::endl;
    // // std::cout << "Profiles:" << std::endl << sb_path.get_profiles() << std::endl;
    std::ostringstream title2;
    title2 << "SimpleBounce: delta = " << delta << ", action = " << sb_path.get_action();
    sb_path.plot_profiles(10., title2.str());
    
    // // Combined plot
    // std::ostringstream title;
    // title << "Bounce path";
    // potential.plot_2d(title.str(), 200, true_vacuum, origin, 0.1, {c2_path});
}