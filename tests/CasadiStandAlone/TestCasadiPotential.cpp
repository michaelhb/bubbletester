#include <casadi/casadi.hpp>
#include <Eigen/Core>
#include "CasadiPotential.hpp"

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

}