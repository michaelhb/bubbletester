#include <casadi/casadi.hpp>
#include <exception>
#include <tuple>

#define sqr(x) (x)*(x)

typedef std::tuple<casadi::SX, casadi::SX> Ca_Potential;

namespace casadi {

std::string varname(std::string prefix, std::vector<int> indices) {
    std::ostringstream ss;
    ss << prefix;
    for (auto &ix : indices) {
        ss << "_" << ix;
    }
    return ss.str();
}

void append_d(std::vector<double> to, std::vector<double> from) {
    to.insert(to.end(), from.begin(), from.end());
}

Ca_Potential get_potential(double delta) {
    SX phi_1 = SX::sym("phi_1");
    SX phi_2 = SX::sym("phi_2");

    SX phi = SX::vertcat(SXVector({phi_1, phi_2}));
    SX V = (sqr(phi_1) + sqr(phi_2))*(1.8*sqr(phi_1 - 1) + 0.2*sqr(phi_2 - 1) - delta);
    return std::make_tuple(V, phi);
}

DM find_false_vac(std::tuple<SX, SX> potential) {
    SX V = std::get<0>(potential);
    SX phi = std::get<1>(potential);

    std::cout << V << std::endl;

    SXDict nlp = {{"x", phi}, {"f", V}};
    Function solver = nlpsol("solver", "ipopt", nlp);
    
    std::vector<double> ub = {2., 2.};
    std::vector<double> lb = {0.5, 0.5};
    std::vector<double> start = {1., 1.};
    
    DMDict args = {{"lbx", lb}, {"ubx", ub}, {"x0", start}};
    DMDict res = solver(args);

    auto it = res.find("x");
    if (it != res.end()) {
        return it->second;
    }
    else {
        throw std::runtime_error("Could not find false vacuum");
    }
}

void solve(Ca_Potential ca_potential, DM false_vac, DM true_vac) {
    SX potential = std::get<0>(ca_potential);
    SX phi = std::get<1>(ca_potential);
    int n_phi = phi.size1();
    
    // Wrap potential as callable function & get value @ true vacuum
    Function fV = Function("fV", {phi}, {potential}, {"phi"}, {"V(phi)"});
    DM v_true = fV(false_vac);

    // Time horizon
    double T = 1.;
    
    // Control intervals and spacing
    int N = 20;
    double h = T/N;

    // Degree of interpolating polynomials
    int d = 3;

    // Linear ansatz
    auto ansatz = [T, false_vac, true_vac](double t) {return (t/T)*false_vac + (1 - t/T)*true_vac;};

    // Derivative of ansatz is a constant
    DM phidot_ansatz = (1/T)*(false_vac - true_vac);

    // Set up the collocation points
    std::vector<double> tau_root = collocation_points(d, "legendre");
    tau_root.insert(tau_root.begin(), 0.);

    // Value of time at point t_k_j
    auto t_kj = [h, tau_root](int k, int j){return h*(k + tau_root[j]);};

    // Coefficients of the collocation equation
    std::vector<std::vector<double> > C(d+1,std::vector<double>(d+1, 0));

    // Coefficients of the continuity equation
    std::vector<double> D(d+1, 0);

    // Coefficients for Gaussian quadrature 
    std::vector<double> B(d+1, 0);

    // Construct polynomial basis & extract relevant coefficients
    for (int j = 0; j < d + 1; ++j) {

        Polynomial p = 1;
        for(int r = 0; r < d + 1; ++r){
            if(r != j){
                p *= Polynomial(-tau_root[r],1)/(tau_root[j]-tau_root[r]);
            }
        }

        // Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0);

        // Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        Polynomial dp = p.derivative();
        for(int r=0; r<d+1; ++r){
            C[j][r] = dp(tau_root[r]);
        }
        
        // Evaluate the integral of the polynomial to get the coefficients for Gaussian quadrature
        Polynomial ip = p.anti_derivative();
        B[j] = ip(1.0);
    }
    
    // Control variables (in this case, corresponding to phi_dot)
    SX u = SX::sym("u", n_phi);

    // Define the integrand in the objective function
    Function L = Function("L", {phi, u}, {sqrt(2*SX::minus(fV(phi)[0], v_true))*norm_2(u)});

    // Dynamics function (trivial here, just u = phidot)
    Function f = Function("f", {phi, u}, {u}, {"phi", "u"}, {"phidot"});

    // Begin constructing NLP
    std::vector<MX> w = {}; // All decision variables
    std::vector<double> w0 = {}; // Initial values for decision variables
    std::vector<double> lbw = {}; // Lower bounds for decision variables
    std::vector<double> ubw = {}; // Upper bounds for decision variables
    std::vector<MX> g = {}; // All constraint functions
    std::vector<double> lbg = {}; // Lower bounds for constraints
    std::vector<double> ubg = {}; // Upper bounds for constraints
    SX J = 0; // Objective function

    // Limits for unbounded variables
    std::vector<double> ubinf(n_phi, inf);
    std::vector<double> lbinf(n_phi, -inf);

    // Initialise control variables
    std::vector<MX> controls = {}; 
    for (int k = 0; k < N; ++k) {
        MX Uk = MX::sym(varname("U", {k}), n_phi);
        controls.push_back(Uk);
        w.push_back(Uk);
        append_d(lbw, lbinf);
        append_d(ubw, ubinf);
        // append_d(w0, phidot_ansatz.get);
    }

}

};

int main() {
    using namespace casadi;

    Ca_Potential potential = get_potential(0.4);
    DM false_vac = find_false_vac(potential);
    DM true_vac = DM::vertcat({0., 0.});
    
    solve(potential, false_vac, true_vac);

    // Function v = get_potential(0.4);
    // std::cout << v << std::endl;
    // DM in = DM::vertcat({1., 1.});
    // std::cout << in << std::endl;
    // DMVector out = v(in);
    // std::cout << out << std::endl;
}