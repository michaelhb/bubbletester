#ifndef BUBBLETESTER_CASADICOLLOCATIONDRIVER2_HPP_INCLUDED
#define BUBBLETESTER_CASADICOLLOCATIONDRIVER2_HPP_INCLUDED

#include <memory>
#include <chrono>
#include <casadi/casadi.hpp>

#include "GenericBounceSolver.hpp"
#include "GenericPotential.hpp"
#include "CasadiCommon.hpp"
#include "CasadiPotential.hpp"
#include "BouncePath.hpp"

namespace BubbleTester {

class CasadiCollocationSolver2 : public GenericBounceSolver {
public:
    CasadiCollocationSolver2(int n_spatial_dimensions_) : n_dims(n_spatial_dimensions_) {
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
            std::cout << "CasadiCollocationSolver2: this is a CasadiPotential" << std::endl;
            Function potential = c_potential.get_function(); 
            return _solve(true_vacuum, false_vacuum, potential);
        }
        catch (const std::bad_cast) {
            // Case 2: we are not working with a CasadiPotential,
            // so we want to wrap it in a Callback and use finite 
            // differences to calculate derivatives.
            std::cout << "CasadiCollocationSolver2: this is not a CasadiPotential" << std::endl;
            CasadiPotentialCallback cb(g_potential);
            Function potential = cb;
            return _solve(true_vacuum, false_vacuum, potential);
        }
    }

    int get_n_spatial_dimensions() const override {
        return n_dims;
    }

    std::string name() override {
        return "CasadiMaupertuis";
    }

    void set_verbose(bool verbose) override {
        // Do nothing for now
    }

    //! Transformation from compact to semi infinite domain
    double Tr(double tau) const {
        return (1.0 + tau) / (1.0 - tau);
    }

    //! Derivative of monotonic transformation
    double Tr_dot(double tau) const {
        return 2.0 / std::pow(tau - 1, 2);
    }

    // //! Inverse of monotonic transformation
    double Tr_inv(double rho) const {
        return (rho - 1.0) / (rho + 1.0);
    }

    //! Derivative of inverse monotonic transformation
    double Tr_inv_dot(double rho) const {
        return 1.0 / (Tr_dot(Tr_inv(rho)));
    }

private:
    int n_dims;

    BouncePath _solve(const Eigen::VectorXd& true_vacuum, 
                      const Eigen::VectorXd& false_vacuum,
                      casadi::Function potential) const {
        using namespace casadi;
        using namespace std::chrono;

        DM true_vac = eigen_to_dm(true_vacuum);
        DM false_vac = eigen_to_dm(false_vacuum);

        std::cout << "FALSE VAC: " << false_vac << std::endl;
        std::cout << "TRUE VAC: " << true_vac << std::endl;

        int n_phi = false_vac.size1();
        SX phi = SX::sym("phi", n_phi);

        // Value of potential at false 
        DM v_true = potential(true_vac);
        
        // Control intervals and spacing (evenly spaced for now)
        int N = 50;
        std::vector<double> t_k;
        std::vector<double> h_k;
        for (int i = 0; i < N; ++i) {
            double t = -1.0 + 2.0*i / N;
            double h = 2.0/N;
            t_k.push_back(t);
            h_k.push_back(h);
        }

        // Degree of interpolating polynomials
        int d = 3;

        // TEMP - until ansatz implemented
        double V0 = -1;

        // Linear ansatz
        auto ansatz = [false_vac, true_vac](double t) {
            return (((t + 1)/2.0)*false_vac + (1.0 - (t + 1.0)/2.0)*true_vac).get_elements();
        };
        // Derivative of ansatz is a constant
        std::vector<double> phidot_ansatz = (false_vac - true_vac).get_elements();

        // Set up the collocation points
        std::vector<double> tau_root = collocation_points(d, "legendre");
        tau_root.insert(tau_root.begin(), 0.);

        // Value of time at point t_k_j
        auto t_kj = [h_k, t_k, tau_root](int k, int j){return t_k[k] + h_k[k]*tau_root[j];};

        // Build vector of parametric inputs (h_k, gamma, gamma_dot)
        std::vector<double> par0;
        par0.insert(par0.end(), h_k.begin(), h_k.end());

        for (int k = 0; k < N; ++k) {
            for (int j = 0; j <= d; ++j) {
                par0.push_back(Tr(t_kj(k,j)));
            }
        }

        for (int k = 0; k < N; ++k) {
            for (int j = 0; j <= d; ++j) {
                par0.push_back(Tr_dot(t_kj(k,j)));
            }
        }

        // Coefficients of the collocation equation
        std::vector<std::vector<double> > C(d+1,std::vector<double>(d+1, 0));

        // Coefficients of the continuity equation
        std::vector<double> D(d+1, 0);

        // Coefficients for Gaussian quadrature 
        std::vector<double> B(d+1, 0);

        // The polynomial basis
        std::vector<Polynomial> P(d + 1);

        // Construct polynomial basis & extract relevant coefficients
        for (int j = 0; j < d + 1; ++j) {

            Polynomial p = 1;
            for(int r = 0; r < d + 1; ++r){
                if(r != j){
                    p *= Polynomial(-tau_root[r],1)/(tau_root[j]-tau_root[r]);
                }
            }
            P[j] = p;

            // Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            D[j] = p(1.0);

            // Evaluate the time derivative of the polynomial at all collocation points 
            Polynomial dp = p.derivative();
            for(int r=0; r<d+1; ++r){
                C[j][r] = dp(tau_root[r]);
            }
            
            // Evaluate the integral of the polynomial to get the coefficients for Gaussian quadrature
            Polynomial ip = p.anti_derivative();
            B[j] = ip(1.0);
        }

        // Begin constructing NLP
        std::vector<SX> w = {}; // All decision variables
        std::vector<double> w0 = {}; // Initial values for decision variables
        std::vector<double> lbw = {}; // Lower bounds for decision variables
        std::vector<double> ubw = {}; // Upper bounds for decision variables
        std::vector<SX> g = {}; // All constraint functions
        std::vector<double> lbg = {}; // Lower bounds for constraints
        std::vector<double> ubg = {}; // Upper bounds for constraints
        std::vector<SX> par = {}; // All parameter variables
        SX T = 0; // Objective function
        SX V = 0; // Integral constraint

        // Limits for unbounded variables
        std::vector<double> ubinf(n_phi, inf);
        std::vector<double> lbinf(n_phi, -inf);

        /**** Initialise parameter variables ****/
        SXVector h_par, gamma_par, gammadot_par;
        for (int i = 0; i < N; ++i) {
            SX h_par_ = SX::sym(varname("h", {i}));
            h_par.push_back(h_par_);
            par.push_back(h_par_);
        }
        for (int i = 0; i < N; ++i) {
            SX gamma_par_ = SX::sym(varname("gamma", {i}), d + 1);
            gamma_par.push_back(gamma_par_);
            par.push_back(gamma_par_);
        }
        for (int i = 0; i < N; ++i) {
            SX gammadot_par_ = SX::sym(varname("gammadot", {i}), d + 1);
            gammadot_par.push_back(gammadot_par_);
            par.push_back(gammadot_par_);
        }

        /**** Initialise control variables ****/
        std::vector<SX> controls = {}; 
        for (int k = 0; k < N + 1; ++k) {
            SX Uk = SX::sym(varname("U", {k}), n_phi);
            controls.push_back(Uk);
            w.push_back(Uk);
            append_d(lbw, lbinf);
            append_d(ubw, ubinf);
            append_d(w0, phidot_ansatz);
        }

        /**** Initialise state variables ****/

        // Start with initial state fixed to true vacuum
        SX phi_0_0 = SX::sym("phi_0_0", n_phi);
        w.push_back(phi_0_0);
        append_d(lbw, true_vac.get_elements());
        append_d(ubw, true_vac.get_elements());
        append_d(w0, true_vac.get_elements());

        // Free endpoint states
        std::vector<SX> endpoints;
        endpoints.push_back(phi_0_0);
        for (int k = 1; k < N; ++k) {
            SX phi_k_0 = SX::sym(varname("phi", {k, 0}), n_phi);
            endpoints.push_back(phi_k_0);
            w.push_back(phi_k_0);
            append_d(lbw, lbinf);
            append_d(ubw, ubinf);
            append_d(w0, ansatz(t_kj(k, 0)));
        }

        // Final state, fixed to the false vacuum
        SX phi_N_0 = SX::sym("phi_N_0", n_phi);
        endpoints.push_back(phi_N_0);
        w.push_back(phi_N_0);
        append_d(lbw, false_vac.get_elements());
        append_d(ubw, false_vac.get_elements());
        append_d(w0, false_vac.get_elements());

        // Build finite elements (including left endpoints)
        std::vector<SXVector> element_states;
        SXVector element_plot;
        for (int k = 0; k < N; ++k) {
            std::vector<SX> e_states;
            e_states.push_back(endpoints[k]);
            for (int j = 1; j <= d; ++j) {
                SX phi_k_j = SX::sym(varname("phi", {k, j}), n_phi);
                e_states.push_back(phi_k_j);
                w.push_back(phi_k_j);
                append_d(lbw, lbinf);
                append_d(ubw, ubinf);
                append_d(w0, ansatz(t_kj(k, j)));
            }
            element_plot.push_back(SX::horzcat(e_states));
            element_states.push_back(e_states);
        }
        
        /**** Useful functions of the state and control variables ****/

        // State variables in a given element
        SXVector element;
        for (int j = 0; j <= d; ++j) {
            element.push_back(SX::sym(varname("phi_k", {j}), n_phi));
        }        

        // Width of control interval
        SX h_elem = SX::sym("h_elem");

        // Monotonic transformation & derivative
        SX gamma = SX::sym("gamma", d + 1);
        SX gammadot = SX::sym("gammadot", d + 1);
        
        // Estimate for the state at end of control interval
        SX phi_end = 0;
        
        for (int i = 0; i <= d; ++i) {
            phi_end += D[i]*element[i];
        }
        
        Function Phi_end = Function("Phi_end", element, {phi_end});

        // Interpolated controls in an element
        SX control_start = SX::sym("u_k", n_phi);
        SX control_end = SX::sym("u_k+1", n_phi);
        SXVector control_int;

        for (int j = 1; j <= d; ++j) {
            control_int.push_back(
                (1 - tau_root[j])*control_start + tau_root[j]*control_end
            );
        }
        
        // Derivative constraints in an element
        SXVector phidot_cons;
        
        for (int j = 1; j <= d; ++j) {
            SX phidot_approx = 0;
            for (int r = 0; r <= d; ++r) {
                phidot_approx += C[r][j]*element[r];
            }
            phidot_cons.push_back(h_elem*gamma(j)*control_int[j - 1] - phidot_approx);
        }
        
        // SXVector phidot_inputs = SXVector(element);
        SXVector phidot_inputs;
        phidot_inputs.insert(phidot_inputs.end(), element.begin(), element.end());
        phidot_inputs.push_back(control_start);
        phidot_inputs.push_back(control_end);
        phidot_inputs.push_back(h_elem);
        phidot_inputs.push_back(gamma);
        
        Function Phidot_cons = Function("Phidot_cons", phidot_inputs, phidot_cons);

        // Value of kinetic objective on an element
        SX T_k = 0;

        for (int j = 1; j <= d; ++j) {
            T_k = T_k + 0.5*h_elem*B[j]*pow(gamma(j), n_dims - 1)
                *gammadot(j)*dot(control_int[j - 1], control_int[j - 1]);
        }
        
        // SXVector quadrature_inputs = SXVector(element);
        SXVector quadrature_inputs;
        quadrature_inputs.insert(quadrature_inputs.end(), element.begin(), element.end());
        quadrature_inputs.push_back(control_start);
        quadrature_inputs.push_back(control_end);
        quadrature_inputs.push_back(h_elem);
        quadrature_inputs.push_back(gamma);
        quadrature_inputs.push_back(gammadot);
        
        Function T_obj_k = Function("T_obj_k", quadrature_inputs, {T_k});
        
        // Value of potential constraint functional on an element
        SX V_k = 0;

        for (int j = 1; j <= d; ++j) {
            V_k = V_k + h_elem*B[j]*pow(gamma(j), n_dims - 1)*gammadot(j)*potential(element[j])[0];
        }

        Function V_cons_k = Function("V_cons_k", quadrature_inputs, {V_k});

        /**** Implement the objective and constraints ****/

        // Zero vector for constraint bounds
        std::vector<double> zeroes(n_phi, 0);

        // Zero vector for collocation bounds
        std::vector<double> zeroes_col(d*n_phi, 0);

        // Continuity equations
        for (int k = 0; k < N; ++k) {
            g.push_back(Phi_end(element_states[k])[0] - endpoints[k + 1]);
            append_d(lbg, zeroes);
            append_d(ubg, zeroes);
        }

        // Collocation equations and objective function
        for (int k = 0; k < N; ++k) {
            SXVector phidot_inputs_ = SXVector(element_states[k]);
            phidot_inputs_.push_back(controls[k]);
            phidot_inputs_.push_back(controls[k + 1]);
            phidot_inputs_.push_back(h_par[k]);
            phidot_inputs_.push_back(gamma_par[k]);

            g.push_back(SX::vertcat(Phidot_cons(phidot_inputs_)));
            append_d(lbg, zeroes_col);
            append_d(ubg, zeroes_col);
        }

        // Build quadratures
        for (int k = 0; k < N; ++k) {
            SXVector quadrature_inputs_ = SXVector(element_states[k]);
            quadrature_inputs_.push_back(controls[k]);
            quadrature_inputs_.push_back(controls[k + 1]);
            quadrature_inputs_.push_back(h_par[k]);
            quadrature_inputs_.push_back(gamma_par[k]);
            quadrature_inputs_.push_back(gammadot_par[k]);

            T += T_obj_k(quadrature_inputs_)[0];
            V += V_cons_k(quadrature_inputs_)[0];
        }

        // Add potential constraint
        g.push_back(V0 - V);
        lbg.push_back(0);
        ubg.push_back(0);

        /**** Initialise and solve the NLP ****/

        // Collect states and constraints into single vectors
        SX W = SX::vertcat(w);
        SX G = SX::vertcat(g);
        SX Par = SX::vertcat(par);

        // Create the solver (this is one of the bottlenecks, so we time it)
        SXDict nlp = {{"f", T}, {"x", W}, {"g", G}, {"p", Par}};
        Dict nlp_opt = Dict();
        nlp_opt["expand"] = false;

        auto t_setup_start = high_resolution_clock::now();
        Function solver = nlpsol("nlpsol", "ipopt", nlp, nlp_opt);
        auto t_setup_end = high_resolution_clock::now();
        auto setup_duration = duration_cast<microseconds>(t_setup_end - t_setup_start).count() * 1e-6;
        std::cout << "CasadiMaupertuisSolver - setup took " << setup_duration << " sec" << std::endl;

        // Run the optimiser. This is the other bottleneck, so we time it too.
        DMDict arg = {{"x0", w0}, {"lbx", lbw}, {"ubx", ubw}, {"lbg", lbg}, {"ubg", ubg}, {"p", par0}};
        auto t_solve_start = high_resolution_clock::now();
        DMDict res = solver(arg);
        auto t_solve_end = high_resolution_clock::now();
        auto solve_duration = duration_cast<microseconds>(t_solve_end - t_solve_start).count() * 1e-6;
        std::cout << "CasadiMaupertuisSolver - optimisation took " << solve_duration << " sec" << std::endl;

        // Return the result (interpolated using the Lagrange representation)
        auto t_extract_start = high_resolution_clock::now(); 
        SX endpoints_plot = SX::horzcat(endpoints);
        SX elements_plot = SX::horzcat(element_plot);

        Function trajectories = Function("trajectories", {W}, {endpoints_plot});
        Function elements = Function("elements", {W}, {elements_plot});

        Eigen::MatrixXd profiles = 
            Eigen::Map<Eigen::MatrixXd>(
                trajectories(res["x"]).at(0).get_elements().data(), n_phi, N + 1).transpose();

        Eigen::MatrixXd elementmx = 
            Eigen::Map<Eigen::MatrixXd>(
                elements(res["x"]).at(0).get_elements().data(), n_phi, N*(d + 1)).transpose();
 
        int points_per = 10;

        Eigen::MatrixXd interpolation = interpolate_elements(elementmx, P, N, n_phi, d, points_per);
        interpolation.conservativeResize(interpolation.rows() + 1, interpolation.cols());
        interpolation.row(interpolation.rows() - 1) = profiles.row(profiles.rows() - 1);

        // Dummy action and radii for now.
        double action = 0;
        Eigen::VectorXd radii = Eigen::VectorXd::Zero(points_per*N + 1);
        
        auto t_extract_end = high_resolution_clock::now();
        auto extract_duration = duration_cast<microseconds>(t_extract_end - t_extract_start).count() * 1e-6;
        std::cout << "CasadiMaupertuisSolver - extracting / formatting took " << extract_duration << " sec" << std::endl;

        std::cout << W.size() << std::endl;

        return BouncePath(radii, interpolation, action);
    }

    Eigen::MatrixXd interpolate_elements(Eigen::MatrixXd elements, std::vector<casadi::Polynomial> P, int n_elem, int n_phi, int d, int points_per) const {
        using namespace casadi;
        
        Eigen::MatrixXd interpolation(n_elem*points_per, n_phi);
        
        double dtau = 1.0/points_per;
        Eigen::VectorXd tau(points_per);
        for (int i = 0; i < points_per; ++i) {
            tau(i) = i*dtau;
        }

        for (int k = 0; k < n_elem; ++k) {
            for (int t = 0; t < points_per; ++t) {
                double tau_t = tau(t);
                Eigen::VectorXd x_k_t = Eigen::VectorXd::Zero(n_phi);

                for (int r = 0; r <= d; ++r) {
                    Eigen::VectorXd x_kr = elements.row(k*(d + 1) + r);
                    x_k_t += P[r](tau_t)*x_kr;
                }
                interpolation.row(k*points_per + t) = x_k_t;
            }
        }
        
        return interpolation;
    }
};

}

#endif