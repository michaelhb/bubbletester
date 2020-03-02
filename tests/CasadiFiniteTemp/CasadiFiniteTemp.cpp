#include <casadi/casadi.hpp>
#include <Eigen/Core>
#include "BouncePath.hpp"
#include "CasadiPotential.hpp"
#include "CasadiMaupertuisDriver.hpp"
#include "SimpleBounceDriver.hpp"
#include "CTFiniteTempModel.hpp"

#define field_vec(x1, x2) (Eigen::VectorXd(2) << x1, x2).finished()

namespace BubbleTester { 

// Print out the value of the potential / derivatives 
// at a few arbitrary sample points
void print_benchmarks() {
    // set up the model
    double m1 = 120.;
    double m2 = 50.;
    double mu = 25.;
    double Y1 = .1;
    double Y2 = .15;
    int n = 30.;
    double renorm_scale = 246.;
    double Tnuc = 84.23819948704514;

    CTFiniteTempModel model = CTFiniteTempModel(m1, m2, mu, Y1, Y2, n, renorm_scale);
    FiniteTempPotential potential = FiniteTempPotential(model, Tnuc);

    std::vector<Eigen::VectorXd> test_points;
    test_points.push_back(field_vec(0., 0.));
    test_points.push_back(field_vec(1., 1.));
    test_points.push_back(field_vec(246., 246.));
    test_points.push_back(field_vec(69.,420.));

    for (auto & phi : test_points) {
        std::cout << "#### test point: " << std::endl << phi << std::endl;
        std::cout << "V(phi) = " << potential(phi) << std::endl;
        std::cout << "dV/dphi_1 = " << potential.partial(phi, 0) << std::endl;
        std::cout << "dV/dphi_2 = " << potential.partial(phi, 1) << std::endl;
        std::cout << "d2V/dphi_1^2 = " << potential.partial(phi, 0, 0) << std::endl;
        std::cout << "d2V/dphi_2^2 = " << potential.partial(phi, 1, 1) << std::endl;
        std::cout << "d2V/dphi_1dphi_2 = " << potential.partial(phi, 0, 1) << std::endl;

        std::cout << std::endl;
    }
}

}

int main() {
    using namespace BubbleTester;    
    print_benchmarks();
}