#include <casadi/casadi.hpp>
#include <Eigen/Core>
#include <chrono>
#include "BouncePath.hpp"
#include "CasadiPotential.hpp"
#include "CasadiCollocationDriver2.hpp"

#define sqr(x) (x)*(x)

std::vector<std::vector<double>> coeffs = {
    {1.8, 0.2},
    {0.684373, 0.181928, 0.295089},
    {0.534808, 0.77023, 0.838912, 0.00517238},
    {0.4747, 0.234808, 0.57023, 0.138912, 0.517238},
    {0.34234, 0.4747, 0.234808, 0.57023, 0.138912, 0.517238},
    {0.5233, 0.34234, 0.4747, 0.234808, 0.57023, 0.138912, 0.517238},
    {0.2434, 0.5233, 0.34234, 0.4747, 0.234808, 0.57023, 0.138912, 0.51723}
};

BubbleTester::CasadiPotential get_potential(int order, double delta0 = 0.4) {
    using namespace casadi;
    using namespace BubbleTester;
    SXVector phi;
    SX delta = SX::sym("delta");
    
    std::vector<double> order_coeffs = coeffs[order - 2];

    for (int i = 0; i < order; ++i) {
        std::ostringstream name;
        name << "phi_" << order;
        phi.push_back(SX::sym(name.str()));
    }

    SX V_term_1 = 0;
    SX V_term_2 = 0;

    for (int i = 0; i < order; ++i) {
        V_term_1 += order_coeffs[i]*sqr(phi[i] - 1);
    }
    V_term_1 -= delta;

    for (int i = 0; i < order; ++i) {
        V_term_2 += sqr(phi[i]);
    }

    SX V = V_term_1*V_term_2;
    std::cout << V << std::endl;
    Function fV = Function("fV", {SX::vertcat(phi), delta}, {V}, {"phi", "delta"}, {"V"});
    return CasadiPotential(fV, order, {delta}, {delta0});
}

Eigen::VectorXd find_true_vac(BubbleTester::CasadiPotential potential) {
    int n_fields = potential.get_number_of_fields();
    Eigen::VectorXd ub(n_fields);
    Eigen::VectorXd lb(n_fields);
    Eigen::VectorXd start(n_fields);

    for (int i = 0; i < n_fields; ++i) {
        ub(i) = 2.;
        lb(i) = 0.5;
        start(i) = 1.;
    }
    return potential.minimise(start, lb, ub);
}

Eigen::VectorXd find_false_vac(BubbleTester::CasadiPotential potential) {
    int n_fields = potential.get_number_of_fields();
    Eigen::VectorXd ub(n_fields);
    Eigen::VectorXd lb(n_fields);
    Eigen::VectorXd start(n_fields);

    for (int i = 0; i < n_fields; ++i) {
        ub(i) = 0.2;
        lb(i) = -0.2;
        start(i) = 0.; 
    }
    return potential.minimise(start, lb, ub);
}

void run_test(int order, double delta_min, double delta_max, int n) {
    using namespace casadi;
    using namespace BubbleTester;
    using namespace std::chrono;

    CasadiPotential potential = get_potential(order);
    std::shared_ptr<GenericBounceSolver> solver = std::make_shared<CasadiCollocationSolver2>(order, 3, 100);

    std::vector<double> deltas(n);
    Eigen::VectorXd origin = Eigen::VectorXd::Zero(order);
    std::vector<Eigen::VectorXd> true_vacua(n);
    std::vector<double> solve_times(n);

    double delta_step = (delta_max - delta_min) / (n - 1);
    for (int i = 0; i < n; ++i) {
        deltas[i] = delta_min + i*delta_step;
        potential.set_param_vals({deltas[i]});
        true_vacua[i] = find_true_vac(potential);
    }

    for (int i = 0; i < n; ++i) {
        auto t_solve_start = high_resolution_clock::now();
        potential.set_param_vals({deltas[i]});
        BouncePath path = solver->solve(true_vacua[i], origin, potential);
        auto t_solve_end = high_resolution_clock::now();
        auto solve_duration = duration_cast<microseconds>(t_solve_end - t_solve_start).count() * 1e-6;
        solve_times[i] = solve_duration;
        std::ostringstream title;
        title << "Collocation solver: delta = " << deltas[i] 
            << ", action = " << path.get_action() << ", t = " << solve_duration << " sec";
        path.plot_profiles(20., title.str());
    }
}

int main() {
    using namespace casadi;
    using namespace BubbleTester;
    using namespace std::chrono;

    run_test(7, 0.01, 0.05, 5);
}
