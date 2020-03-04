#include <Eigen/Core>
#include <iostream>
#include <sys/time.h>
#include <stdlib.h>
#include <vector> 

#include "GenericBounceSolver.hpp"
#include "BP1Driver.hpp"
#include "SimpleBounceDriver.hpp"
#include "BouncePath.hpp"
#include "CTFiniteTempModel.hpp"
#include "CasadiMaupertuisDriver.hpp"

typedef std::tuple<double, double> point_marker;

int n_spatial_dimensions = 3;

namespace BubbleTester {

struct vevs {
    Eigen::VectorXd true_vacuum;
    Eigen::VectorXd false_vacuum;
};

vevs find_vacua(CTFiniteTempModel model, double T) {
    FiniteTempPotential potential = FiniteTempPotential(model, T);
    
    Eigen::VectorXd lb_f(2), lb_t(2), ub_f(2), ub_t(2), 
        start_f(2), start_t(2), true_vacuum(2), false_vacuum(2);
    
    lb_f << 0, -600;
    ub_f << 600, 0;
    lb_t << 0, 0;
    ub_t << 600, 600;
    start_f << 250, -250;
    start_t << 250, 250;

    true_vacuum = potential.minimise(start_t, lb_t, ub_t);
    false_vacuum = potential.minimise(start_f, lb_f, ub_f);
    return vevs{true_vacuum, false_vacuum};
}
};

int main() {
    using namespace BubbleTester;
    
    double renorm_scale = 246.;
    
    // cosmotransitions version
    // double m1 = 120.;
    // double m2 = 50.;
    // double mu = 25.;
    // double Y1 = .1;
    // double Y2 = .15;
    // int n = 30.;
    
    // Nice curved case
    // double m1 = 120.;
    // double m2 = 50.;
    // double mu = 2.;
    // double Y1 = 1.;
    // double Y2 = .15;
    // int n = 30.;
    // double T = 78.;

    double m1 = 120.;
    double m2 = 50.;
    double mu = 25.;
    double Y1 = 0.5;
    double Y2 = .15;
    int n = 30.;
    double T = 25.;

    CTFiniteTempModel model = CTFiniteTempModel(m1, m2, mu, Y1, Y2, n, renorm_scale);
    
    FiniteTempPotential potential(model, T);
    std::ostringstream title;

    vevs vacua = find_vacua(model, T);
    std::cout << "True vacuum:" << std::endl << vacua.true_vacuum << std::endl;
    std::cout << "False vacuum:" << std::endl << vacua.false_vacuum << std::endl;

    std::shared_ptr<GenericBounceSolver> bp_solver = std::make_shared<BP1BounceSolver>(n_spatial_dimensions);
    std::shared_ptr<GenericBounceSolver> sb_solver = std::make_shared<SimpleBounceSolver>(1., 100., n_spatial_dimensions);
    std::shared_ptr<GenericBounceSolver> mp_solver = std::make_shared<CasadiMaupertuisSolver>(n_spatial_dimensions);

    bp_solver->set_verbose(true);
    sb_solver->set_verbose(true);
    mp_solver->set_verbose(true);

    std::vector<BouncePath> paths;

    bool normalise = true;
    double margin = 50.;

    Eigen::VectorXd true_vacuum_n(2), false_vacuum_n(2);
    true_vacuum_n = vacua.true_vacuum;
    false_vacuum_n = vacua.false_vacuum;

    if (normalise) {
        GenericPotential::normalise(potential, n_spatial_dimensions, vacua.true_vacuum, vacua.false_vacuum);    
        true_vacuum_n << 1., 0.;
        false_vacuum_n << 0., 0.;
        margin = .2;
    }


    std::cout << "v(true) = " << potential(true_vacuum_n) << std::endl;
    std::cout << "v(false) = " << potential(false_vacuum_n) << std::endl;
    std::cout << "dV/dx1 (false) = " << potential.partial(false_vacuum_n, 0) << std::endl;
    std::cout << "dV/dx2 (false) = " << potential.partial(false_vacuum_n, 1) << std::endl;
    std::cout << "dV/dx1 (true) = " << potential.partial(true_vacuum_n, 0) << std::endl;
    std::cout << "dV/dx2 (true) = " << potential.partial(true_vacuum_n, 1) << std::endl;
    
    paths.push_back(mp_solver->solve(true_vacuum_n, false_vacuum_n, potential));
    potential.plot_2d(title.str(), 200, true_vacuum_n, false_vacuum_n, margin, paths);
}

