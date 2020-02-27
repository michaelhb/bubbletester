#include <casadi/casadi.hpp>
#include <Eigen/Core>
#include "BouncePath.hpp"
#include "CasadiPotential.hpp"
#include "CasadiMaupertuisDriver.hpp"
#include "BP1Driver.hpp"
#include "SimpleBounceDriver.hpp"

#define sqr(x) (x)*(x)

casadi::Function get_potential(double delta) {
    using namespace casadi;
    SX phi_1 = SX::sym("phi_1");
    SX phi_2 = SX::sym("phi_2");

    SX phi = SX::vertcat(SXVector({phi_1, phi_2}));
    SX V = (sqr(phi_1) + sqr(phi_2))*(1.8*sqr(phi_1 - 1) + 0.2*sqr(phi_2 - 1) - delta);
    return Function("fV", {phi}, {V}, {"phi"}, {"V(phi)"});
}

int main() {
    // using namespace casadi;
    using namespace BubbleTester;
    casadi::Function fPotential = get_potential(0.4);
    CasadiPotential potential = CasadiPotential(fPotential, 2);

    // Eigen::VectorXd test1(2);
    // Eigen::VectorXd test2(2);
    // Eigen::VectorXd test3(2);
    // Eigen::VectorXd test4(2);
    // test1 << 0., 0.;
    // test2 << 1., 1.;
    // test3 << 3., -3.;
    // test4 << 4., 2.;
    // std::vector<Eigen::VectorXd> tests = {test1, test2, test3, test4};
    // for (int i = 0; i < 4; ++i) {
    //     std::cout << "Test " << i << " ####" << std::endl;
    //     std::cout << "V(phi): " << potential(tests[i]) << std::endl;
    //     std::cout << "grad: ";
    //     std::cout << potential.partial(tests[i], 0) << " ";
    //     std::cout << potential.partial(tests[i], 1) << std::endl;
    //     std::cout << "hess: ";
    //     std::cout << potential.partial(tests[i], 0, 0) << " ";
    //     std::cout << potential.partial(tests[i], 0, 1) << " ";
    //     std::cout << potential.partial(tests[i], 1, 0) << " ";
    //     std::cout << potential.partial(tests[i], 1, 1) << std::endl;
    // }
    
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

    // Find the thin wall limit solution
    std::shared_ptr<GenericBounceSolver> mp_solver = std::make_shared<CasadiMaupertuisSolver>(2);
    BouncePath mp_path = mp_solver->solve(true_vacuum, origin, potential);
    potential.plot_2d("CasADi thin wall limit", 200, true_vacuum, origin, 0.5, mp_path);

    // Have a go with BubbleProfiler
    // std::shared_ptr<GenericBounceSolver> bp_solver = std::make_shared<BP1BounceSolver>(2);
    // bp_solver->set_verbose(true);
    // BouncePath bp_path = bp_solver->solve(true_vacuum, origin, potential);
    // potential.plot_2d("BubbleProfiler solution", 200, true_vacuum, origin, 0.5, bp_path);

    // Have a go with SimpleBounce
    std::shared_ptr<GenericBounceSolver> sb_solver = std::make_shared<SimpleBounceSolver>(1., 100., 3);
    sb_solver->set_verbose(true);
    BouncePath sb_path = sb_solver->solve(true_vacuum, origin, potential);
    std::cout << "Action = " << sb_path.get_action() << std::endl;
    potential.plot_2d("SimpleBounce solution", 200, true_vacuum, origin, 0.5, sb_path);
    
}