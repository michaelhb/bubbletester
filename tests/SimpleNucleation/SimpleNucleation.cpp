#include <casadi/casadi.hpp>
#include <Eigen/Core>
#include <chrono>
#include "BouncePath.hpp"
#include "CasadiPotential.hpp"
#include "CasadiCollocationDriver2.hpp"

#define sqr(x) (x)*(x)


int main() {
    using namespace casadi;
    using namespace BubbleTester;
    
    SX delta = SX::sym("delta");
    double delta0 = 0.04;

    SX phi_1 = SX::sym("phi_1");
    SX phi_2 = SX::sym("phi_2");
    
    SX V = (sqr(phi_1) + sqr(phi_2))*(1.8*sqr(phi_1 - 1) + 0.2*sqr(phi_2 - 1) - delta);

    SX phi = SX::vertcat(SXVector({phi_1, phi_2}));
    Function fPotential = Function("fV", {phi, delta}, {V}, {"phi", "delta"}, {"V"});

    CasadiPotential potential = CasadiPotential(fPotential, 2, {delta}, {delta0});

    // Find the location of the true vacuum
    Eigen::VectorXd origin = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd ub(2);
    Eigen::VectorXd lb(2);
    Eigen::VectorXd start(2);
    ub << 2., 2.;
    lb << .5, .5;
    start << 1., 1.;
    
    Eigen::VectorXd true_vacuum = potential.minimise(start, lb, ub);
    std::cout << "True vacuum:" << std::endl << true_vacuum << std::endl;

    std::shared_ptr<GenericBounceSolver> c2_solver = std::make_shared<CasadiCollocationSolver2>(2, 3, 100);
    c2_solver->set_verbose(true);
    BouncePath c2_path = c2_solver->solve(true_vacuum, origin, potential);
    std::cout << "Action = " << c2_path.get_action() << std::endl;
    c2_path.plot_profiles(20., "Collocation Solver 2");
}
