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

void append_d(std::vector<double> &to, std::vector<double> from) {
    // Yes I know this is a weird function signature :)
    to.insert(to.end(), from.begin(), from.end());
}

// Our test potential, using CasADi primitives
Function get_potential(double delta) {
    SX phi_1 = SX::sym("phi_1");
    SX phi_2 = SX::sym("phi_2");

    SX phi = SX::vertcat(SXVector({phi_1, phi_2}));
    SX V = (sqr(phi_1) + sqr(phi_2))*(1.8*sqr(phi_1 - 1) + 0.2*sqr(phi_2 - 1) - delta);
    return Function("fV", {phi}, {V}, {"phi"}, {"V(phi)"});
}

// Same thing, but as a black box Callback function
class PotentialCallback : public Callback {
public:
    PotentialCallback(double delta_) : delta(delta_) {
        Dict opts;
        opts["enable_fd"] = true;
        construct("cV", opts);
    }

    casadi_int get_n_in() override { return 1; }
    casadi_int get_n_out() override { return 1; }

    Sparsity get_sparsity_in(casadi_int i) {
        return Sparsity::dense(2, 1);
    }

    std::vector<DM> eval(const std::vector<DM>& arg) const override {
        DM phi_1 = arg.at(0).get_elements()[0];
        DM phi_2 = arg.at(0).get_elements()[1];
        return {(sqr(phi_1) + sqr(phi_2))*(1.8*sqr(phi_1 - 1) + 0.2*sqr(phi_2 - 1) - delta)};
    }

private:
    double delta;
};

DM find_false_vac(Function V, int n_phi) {
    MX phi = MX::sym("phi", n_phi);
    
    MX sV = V(phi)[0];

    MXDict nlp = {{"x", phi}, {"f", sV}};
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

void solve(Function potential, DM false_vac, DM true_vac) {\
    int n_phi = false_vac.size1();
    SX phi = SX::sym("phi", n_phi);
    
    // Value of potential at false 
    DM v_true = potential(false_vac);

    // Time horizon
    double T = 1.;
    
    // Control intervals and spacing
    int N = 20;
    double h = T/N;

    // Degree of interpolating polynomials
    int d = 3;

    // Linear ansatz
    auto ansatz = [T, false_vac, true_vac](double t) {
        return ((t/T)*false_vac + (1 - t/T)*true_vac).get_elements();
    };

    // Derivative of ansatz is a constant
    std::vector<double> phidot_ansatz = ((1/T)*(false_vac - true_vac)).get_elements();

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
    Function L = Function("L", 
        {phi, u}, {sqrt(2*SX::minus(potential(phi)[0], v_true))*norm_2(u)},
        {"phi", "u"}, {"L(phi, u)"});
    
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
    MX J = 0; // Objective function

    // Limits for unbounded variables
    std::vector<double> ubinf(n_phi, inf);
    std::vector<double> lbinf(n_phi, -inf);

    /**** Initialise control variables ****/
    std::vector<MX> controls = {}; 
    for (int k = 0; k < N; ++k) {
        MX Uk = MX::sym(varname("U", {k}), n_phi);
        controls.push_back(Uk);
        w.push_back(Uk);
        append_d(lbw, lbinf);
        append_d(ubw, ubinf);
        append_d(w0, phidot_ansatz);
    }

    /**** Initialise state variables ****/

    // Start with initial state fixed to true vacuum
    MX phi_0_0 = MX::sym("phi_0_0", n_phi);
    w.push_back(phi_0_0);
    append_d(lbw, true_vac.get_elements());
    append_d(ubw, true_vac.get_elements());
    append_d(w0, true_vac.get_elements());

    // Free endpoint states
    std::vector<MX> endpoints;
    endpoints.push_back(phi_0_0);
    for (int k = 1; k < N; ++k) {
        MX phi_k_0 = MX::sym(varname("phi", {k, 0}), n_phi);
        endpoints.push_back(phi_k_0);
        w.push_back(phi_k_0);
        append_d(lbw, lbinf);
        append_d(ubw, ubinf);
        append_d(w0, ansatz(t_kj(k, 0)));
    }

    // Final state, fixed to the false vacuum
    MX phi_N_0 = MX::sym("phi_N_0", n_phi);
    endpoints.push_back(phi_N_0);
    w.push_back(phi_N_0);
    append_d(lbw, false_vac.get_elements());
    append_d(ubw, false_vac.get_elements());
    append_d(w0, false_vac.get_elements());

    // Intermediate free states at collocation points
    std::vector<std::vector<MX>> collocation_states;
    for (int k = 0; k < N; ++k) {
        std::vector<MX> k_states;
        for (int j = 1; j <= d; ++j) {
            MX phi_k_j = MX::sym(varname("phi", {k, j}), n_phi);
            k_states.push_back(phi_k_j);
            w.push_back(phi_k_j);
            append_d(lbw, lbinf);
            append_d(ubw, ubinf);
            append_d(w0, ansatz(t_kj(k, j)));
        }
        collocation_states.push_back(k_states);
    }

    /**** Implement the constraints ****/

    // Zero vector for constraint bounds
    std::vector<double> zeroes(n_phi, 0);

    // Continuity equations
    for (int k = 0; k < N; ++k) {
        // Approximation of phi_(k+1)_0 using the k-domain collocation points
        MX phi_end = D[0]*endpoints[k];
        for (int j = 1; j <= d; ++j) {
            phi_end += D[j]*collocation_states[k][j - 1];
        }

        // We require that this is equal to the the actual value of phi_(k+1)_0
        g.push_back(phi_end - endpoints[k + 1]);
        append_d(lbg, zeroes);
        append_d(ubg, zeroes);
    }

    // Collocation equations
    for (int k = 0; k < N; ++k) {
        for (int j = 1; j <= d; ++j) {
            // Approximation of the state derivative at each collocation point
            MX phidot_approx = C[0][j]*endpoints[k];
            for (int r = 0; r < d; ++r) {
                phidot_approx += C[r + 1][j]*collocation_states[k][r];
            }

            // We relate this to the derivative (control) U_k
            g.push_back(h*controls[k] - phidot_approx);
            append_d(lbg, zeroes);
            append_d(ubg, zeroes);
        }
    }

    /**** Construct the objective function ****/
    for (int k = 0; k < N; ++k) {
        for (int j = 1; j <= d; ++j) {
            std::vector<MX> arg = {collocation_states[k][j - 1], controls[k]};
            MX dL = L(arg)[0];
            J = J + B[j]*dL*h;
        }
    }

    /**** Initialise and solve the NLP ****/
    
    // Collect states and constraints into single vectors
    MX W = MX::vertcat(w);
    MX G = MX::vertcat(g);

    // Create the solver
    MXDict nlp = {{"f", J}, {"x", W}, {"g", G}};
    Function solver = nlpsol("nlpsol", "ipopt", nlp);

    DMDict arg = {{"x0", w0}, {"lbx", lbw}, {"ubx", ubw}, {"lbg", lbg}, {"ubg", ubg}};

    DMDict res = solver(arg);

    MX endpoints_plot = MX::horzcat(endpoints);
    MX controls_plot =  MX::horzcat(controls);

    Function trajectories = Function("trajectories", {W}, {endpoints_plot, controls_plot});
    std::cout << trajectories(res["x"]) << std::endl;
    std::cout << endpoints_plot << std::endl;
}

};

int main() {
    using namespace casadi;
    Function potential = get_potential(0.4);
    PotentialCallback cb_potential(0.4);

    DM false_vac = find_false_vac(potential, 2);
    DM true_vac = DM::vertcat({0., 0.});


    solve(potential, false_vac, true_vac);   
}