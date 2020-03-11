#ifndef BUBBLETESTER_CASADIMAUPERTUISDRIVER_HPP_INCLUDED
#define BUBBLETESTER_CASADIMAUPERTUISDRIVER_HPP_INCLUDED

#include <memory>
#include <casadi/casadi.hpp>

#include "GenericBounceSolver.hpp"
#include "GenericPotential.hpp"
#include "CasadiPotential.hpp"
#include "BouncePath.hpp"


namespace BubbleTester {

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

class CasadiMaupertuisSolver : public GenericBounceSolver {
public:
    CasadiMaupertuisSolver(int n_spatial_dimensions_) : n_spatial_dimensions(n_spatial_dimensions_) {
    }

    BouncePath solve(
        const Eigen::VectorXd& true_vacuum,
        const Eigen::VectorXd& false_vacuum,
        const GenericPotential& g_potential) const override {
        using namespace casadi;

        bool is_casadi = false;

        // Hacky way of figuring out if it's a CasadiPotential 
        try {
            // Case 1: we are working with a CasadiPotential, 
            // so we want to use the Function instance. 
            const CasadiPotential &c_potential = dynamic_cast<const CasadiPotential &>(g_potential);
            std::cout << "CasadiMaupertuisSolver: this is a CasadiPotential" << std::endl;
            Function potential = c_potential.get_function(); 
            return _solve(true_vacuum, false_vacuum, potential);
        }
        catch (const std::bad_cast) {
            // Case 2: we are not working with a CasadiPotential,
            // so we want to wrap it in a Callback and use finite 
            // differences to calculate derivatives.
            std::cout << "CasadiMaupertuisSolver: this is not a CasadiPotential" << std::endl;
            CasadiPotentialCallback cb(g_potential);
            Function potential = cb;
            return _solve(true_vacuum, false_vacuum, potential);
        }
    }

    int get_n_spatial_dimensions() const override {
        return n_spatial_dimensions;
    }

    std::string name() override {
        return "CasadiMaupertuis";
    }

    void set_verbose(bool verbose) override {
        // Do nothing for now
    }

private:
    int n_spatial_dimensions;

    BouncePath _solve(const Eigen::VectorXd& true_vacuum, 
                      const Eigen::VectorXd& false_vacuum,
                      casadi::Function potential) const {
        using namespace casadi;

        DM true_vac = eigen_to_dm(true_vacuum);
        DM false_vac = eigen_to_dm(false_vacuum);

        std::cout << "FALSE VAC: " << false_vac << std::endl;
        std::cout << "TRUE VAC: " << true_vac << std::endl;

        int n_phi = false_vac.size1();
        MX phi = MX::sym("phi", n_phi);

        // Value of potential at false 
        DM v_true = potential(true_vac);

        // Time horizon
        double T = 1.;
        
        // Control intervals and spacing
        int N = 50;
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
        MX u = MX::sym("u", n_phi);

        // Define the integrand in the objective function
        Function L = Function("L", 
            {phi, u}, {sqrt(2*MX::abs(MX::minus(potential(phi)[0], v_true)))*norm_2(u)},
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
        for (int k = 0; k < N + 1; ++k) {
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

                // Linear interpolation of controls
                MX control_int = tau_root[j]*controls[k + 1] + (1 - tau_root[j])*controls[k];

                // We relate this to the derivative (control) U_k
                g.push_back(h*control_int - phidot_approx);
                append_d(lbg, zeroes);
                append_d(ubg, zeroes);
            }
        }

        /**** Construct the objective function ****/
        for (int k = 0; k < N; ++k) {
            for (int j = 1; j <= d; ++j) {
                // Linear interpolation of controls
                MX control_int = tau_root[j]*controls[k + 1] + (1 - tau_root[j])*controls[k];
                
                std::vector<MX> arg = {collocation_states[k][j - 1], control_int};
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

        Eigen::MatrixXd profiles = 
            Eigen::Map<Eigen::MatrixXd>(
                trajectories(res["x"]).at(0).get_elements().data(), n_phi, N + 1).transpose();

        Eigen::VectorXd radii(N + 1);
        for (int k = 0; k <= N; ++k) {
            radii(k) = t_kj(k, 0);
        }

        // We're not calculating the action yet, set it to zero            
        double action = 0;

        return BouncePath(radii, profiles, action);
    }
};

}

#endif