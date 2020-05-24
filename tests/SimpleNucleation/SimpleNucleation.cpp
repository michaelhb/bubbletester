#include <casadi/casadi.hpp>
#include <Eigen/Core>
#include <chrono>
#include "BouncePath.hpp"
#include "CasadiPotential.hpp"
#include "CasadiCollocationDriver2.hpp"

#define sqr(x) (x)*(x)

std::vector<std::vector<double>> true_vacua = {
    {1.002686501,1.024591923},
    {1.005159378,1.048428535},
    {1.00747025,1.071525037},
    {1.009628624,1.093959436},
    {1.011663556,1.115781963},
    {1.013579369,1.136995792},
    {1.015377283,1.157680154},
    {1.017056584,1.177906454},
    {1.018675804,1.197568417},
    {1.020196021,1.216833979},
    {1.021665573,1.235630989},
    {1.023027778,1.254077256},
    {1.024352551,1.272139311},
    {1.025596261,1.289813697},
    {1.026814461,1.307156086},
    {1.027971268,1.324152946},
    {1.029055953,1.340857327},
    {1.03013289,1.357264698},
    {1.031151533,1.373384356},
    {1.032126427,1.389195919},
    {1.033080518,1.404829651},
    {1.033984542,1.420169055},
    {1.034877777,1.435247421},
    {1.035737038,1.450214863},
    {1.036540985,1.464780807},
    {1.03734833,1.479306668},
    {1.038118601,1.493562102},
    {1.038868427,1.507617712},
    {1.039605618,1.521497011}};

std::vector<double> deltas = {0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29};

int main() {
    using namespace casadi;
    using namespace BubbleTester;
    using namespace std::chrono;
    
    // Create the potential 
    SX delta = SX::sym("delta");
    double delta0 = 0.01;

    SX phi_1 = SX::sym("phi_1");
    SX phi_2 = SX::sym("phi_2");
    
    SX V = (sqr(phi_1) + sqr(phi_2))*(1.8*sqr(phi_1 - 1) + 0.2*sqr(phi_2 - 1) - delta);

    SX phi = SX::vertcat(SXVector({phi_1, phi_2}));
    Function fPotential = Function("fV", {phi, delta}, {V}, {"phi", "delta"}, {"V"});

    CasadiPotential potential = CasadiPotential(fPotential, 2, {delta}, {delta0});

    // // Solve the bounces

    std::cout << std::setprecision(10);
    std::shared_ptr<GenericBounceSolver> c2_solver = std::make_shared<CasadiCollocationSolver2>(2, 3, 100);
    Eigen::VectorXd origin = Eigen::VectorXd::Zero(2);
    
    for (int i = 0; i < deltas.size(); ++i) {
        potential.set_param_vals({deltas[i]});
        Eigen::VectorXd true_vacuum(2);
        true_vacuum << true_vacua[i][0], true_vacua[i][1];
        
        auto t_solve_start = high_resolution_clock::now();
        BouncePath c2_path = c2_solver->solve(true_vacuum, origin, potential);
        auto t_solve_end = high_resolution_clock::now();
        auto solve_duration = duration_cast<microseconds>(t_solve_end - t_solve_start).count() * 1e-6;
        std::cout << "Found bounce in " << solve_duration << " sec" << std::endl;

        std::ostringstream title;
        title << "Collocation solver 2: delta = " << deltas[i] << 
            ", action = " << c2_path.get_action() << ", t = " << solve_duration << " sec";

        c2_path.plot_profiles(15., title.str());
    }

    // Find the location of the true vacuum
    // Eigen::VectorXd origin = Eigen::VectorXd::Zero(2);
    // Eigen::VectorXd ub(2);
    // Eigen::VectorXd lb(2);
    // Eigen::VectorXd start(2);
    // ub << 2., 2.;
    // lb << .5, .5;
    // start << 1., 1.;

    // double delta_min = 0.01;
    // double delta_max = 0.3;
    // double delta_step = 0.01;
    
    // double delta_ = delta_min;

    // std::vector<double> deltas;
    // std::vector<Eigen::VectorXd> true_vacua;

    // do {
    //     std::cout << delta_ << std::endl;
    //     deltas.push_back(delta_);
    //     potential.set_param_vals({delta_});
    //     true_vacua.push_back(potential.minimise(start, lb, ub));
    //     delta_ += delta_step;
    // }
    // while (delta_< delta_max);

    // std::cout << "std::vector<std::vector<double>> true_vacua = {{";
    // for (auto & vacuum : true_vacua) {
    //     std::cout << vacuum(0) << "," << vacuum(1) << "},{";
    // }
    // std::cout << "}};" << std::endl;

    // std::cout << "std::vector<double> deltas = {";
    // for (auto & delta_ : deltas) {
    //     std::cout << delta_ << ",";
    // }
    // std::cout << "};" << std::endl;

    
    // Eigen::VectorXd true_vacuum = potential.minimise(start, lb, ub);
    // std::cout << "True vacuum:" << std::endl << true_vacuum << std::endl;

    // std::shared_ptr<GenericBounceSolver> c2_solver = std::make_shared<CasadiCollocationSolver2>(2, 3, 100);
    // c2_solver->set_verbose(true);

    // BouncePath c2_path = c2_solver->solve(true_vacuum, origin, potential);
    // std::cout << "Action = " << c2_path.get_action() << std::endl;
    // c2_path.plot_profiles(15., "Collocation Solver 2");

    // potential.set_param_vals({0.5});

    // true_vacuum = potential.minimise(start, lb, ub);
    // std::cout << "True vacuum:" << std::endl << true_vacuum << std::endl;

    // BouncePath c2_path2 = c2_solver->solve(true_vacuum, origin, potential);
    // std::cout << "Action = " << c2_path2.get_action() << std::endl;
    // c2_path2.plot_profiles(5., "Collocation Solver 2");
}
