#include <casadi/casadi.hpp>
#include <Eigen/Core>
#include "CasadiPotential.hpp"
#define sqr(x) (x)*(x)

// All temp
namespace casadi {

Function get_potential(double delta) {
    SX phi_1 = SX::sym("phi_1");
    SX phi_2 = SX::sym("phi_2");

    SX phi = SX::vertcat(SXVector({phi_1, phi_2}));
    SX V = (sqr(phi_1) + sqr(phi_2))*(1.8*sqr(phi_1 - 1) + 0.2*sqr(phi_2 - 1) - delta);
    return Function("fV", {phi}, {V}, {"phi"}, {"V(phi)"});
}

DMVector eigen_to_dmvec(Eigen::VectorXd vec) {
    DMVector dmVec;
    for (int i = 0; i < vec.size(); ++i) {
        dmVec.push_back(vec(i));
    }
    return dmVec;
}

}

int main() {
    using namespace casadi;
    using namespace BubbleTester;
    Function fPotential = get_potential(0.4);
    CasadiPotential potential = CasadiPotential(fPotential, 2);

    Eigen::VectorXd arg(2);
    arg << 1.0, 1.5;

    std::cout << potential(arg) << std::endl;
    std::cout << potential.partial(arg, 0) << std::endl;
    std::cout << potential.partial(arg, 0, 1) << std::endl;

    // MX phi = MX::sym("phi", 2);

    // MX grad = gradient(MX::vertcat(potential(phi)), phi);
    // Function fGrad = Function("fGrad", {phi}, {grad});

    // MX hessian = MX::hessian(MX::vertcat(potential(phi)), phi);
    // Function fHessian = Function("fHessian", {phi}, {hessian});
    
    // Eigen::VectorXd eArg(2);
    // eArg << 0., 0.9;
    
    // DM arg = DM::vertcat(eigen_to_dmvec(eArg));

    // DMVector rGrad = fGrad(arg);
    // DM rHess = fHessian(arg)[0];
    
    // DM rHess00 = rHess(0,0);
    // double dHess00 = (double) rHess00;
    // DM grad0 = rGrad[0](0);
    // double dgrad0 = (double) grad0;
    // std::cout << rHess << std::endl;
    // std::cout << dHess00 << std::endl;
    // std::cout << rGrad << std::endl;
    // std::cout << dgrad0 << std::endl;
}