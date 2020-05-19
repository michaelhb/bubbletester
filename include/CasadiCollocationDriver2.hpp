#ifndef BUBBLETESTER_CASADICOLLOCATIONDRIVER2_HPP_INCLUDED
#define BUBBLETESTER_CASADICOLLOCATIONDRIVER2_HPP_INCLUDED

#include <memory>
#include <chrono>
#include <exception>
#include <casadi/casadi.hpp>

#include "GenericBounceSolver.hpp"
#include "GenericPotential.hpp"
#include "CasadiCommon.hpp"
#include "CasadiPotential.hpp"
#include "BouncePath.hpp"

namespace BubbleTester {

struct Ansatz {
    double V0;
    std::vector<double> Phi0;
    std::vector<double> U0;
};

class CasadiCollocationSolver2 : public GenericBounceSolver {
public:
    CasadiCollocationSolver2(int n_spatial_dimensions_, int N_) : n_dims(n_spatial_dimensions_), N(N_), d(3) {
        tau_root = casadi::collocation_points(d, "legendre");
        tau_root.insert(tau_root.begin(), 0.);
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

private:
    int n_dims; // Number of spatial dimensions
    int N; // Number of finite elements
    int d; // Degree of interpolating polynomials

    mutable std::vector<double> t_k; // Element start times
    mutable std::vector<double> h_k; // Element widths

    // Vector of collocation points on [0,1)
    std::vector<double> tau_root;

    // Time at element k, collocation point j
    double t_kj(int k, int j) const {
        return t_k[k] + h_k[k]*tau_root[j];
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

    //! Ansatz in semi-infinite coordinates
    casadi::DM ansatz(double rho, casadi::DM true_vac, casadi::DM false_vac, 
        double r0, double sigma) const {        
        return true_vac + 0.5*(false_vac - true_vac)*(1 
            + tanh((rho - r0) / sigma)
            + exp(-rho)/(sigma*std::pow(cosh(r0/sigma),2)));
    }

    //! Derivative of ansatz in semi-infinite coordinates
    casadi::DM ansatz_dot(double rho, casadi::DM true_vac, casadi::DM false_vac,
        double r0, double sigma) const {
        return ((false_vac - true_vac)/2.0)*(
                1.0/(sigma*std::pow(cosh((rho-r0)/sigma), 2)) -
                exp(-rho)/(sigma*std::pow(cosh(r0/sigma), 2)));
    }

    Ansatz get_ansatz(casadi::Function fV, 
        std::vector<double> par0, casadi::DM true_vac, casadi::DM false_vac) const {

        Ansatz a;
        std::vector<double> Phi0, U0;
        Phi0.reserve((N*(n_dims + 1) + 1));
        U0.reserve(N + 1);
        casadi::DMDict argV;
        argV["par"] = par0;

        double r0 = 2.5;
        double targetV = -1;
        double tol = 1e-3;
        double sig_upper = 6.;
        double sig_lower = 0.;
        double sigma = 0.5*(sig_upper + sig_lower);
        double sigma_cache = sigma;
        double V_mid;
        
        do {
            Phi0.clear();

            // Endpoints
            for (int k = 0; k <= N; ++k) {
                append_d(Phi0, ansatz(
                    Tr(t_kj(k, 0)), true_vac, false_vac, r0, sigma).get_elements());
            }

            // Collocation points
            for (int k = 0; k < N; ++k) {
                for (int j = 1; j <= d; ++j) {
                    append_d(Phi0, ansatz(
                        Tr(t_kj(k, j)), true_vac, false_vac, r0, sigma ).get_elements());
                }
            }

            argV["Phi"] = Phi0;
            V_mid = fV(argV).at("V").get_elements()[0];

            std::cout << "r0 = " << r0 << ", sigma = " << sigma 
                      << ", V_mid = " << V_mid << std::endl; 

            if (V_mid < targetV) {
                sig_lower  = sigma;
            }
            else if (V_mid > targetV) {
                sig_upper = sigma;
            }
            sigma_cache = sigma;
            sigma = 0.5*(sig_lower + sig_upper);
        }
        while (std::abs(V_mid - targetV) > tol);

        // Calculate the derivatives
        for (int k = 0; k < N; ++k) {
            append_d(U0, ansatz_dot(
                Tr(t_kj(k,0)), true_vac, false_vac, r0, sigma_cache).get_elements());
        }

        // Avoid Tr(1) singularity
        append_d(U0, std::vector<double>(false_vac.size1(), 0));

        a.Phi0 = Phi0;
        a.U0 = U0;
        a.V0 = V_mid;
        return a;
    }
    
    BouncePath _solve(const Eigen::VectorXd& true_vacuum, 
                      const Eigen::VectorXd& false_vacuum,
                      casadi::Function potential) const {
        using namespace casadi;
        using namespace std::chrono;

        DM true_vac = eigen_to_dm(true_vacuum);
        DM false_vac = eigen_to_dm(false_vacuum);

        std::cout << "FALSE VAC: " << false_vac << std::endl;
        std::cout << "TRUE VAC: " << true_vac << std::endl;

         // TEMP - hard coded ansatz parameters (r0 = 2., sigma=.5 good for delta = 0.4)
        double r0 = 4;
        double sigma = 2;

        // TEMP - hard coded volume factors
        double S_n;
        if (n_dims == 3) {
            S_n = 4*pi;
        }
        else if (n_dims == 4) {
            S_n = 0.5*pi*pi;
        }
        else {
            throw std::invalid_argument("Only d = 3 and d = 4 are currently supported.");
        }

        int n_phi = false_vac.size1();
        SX phi = SX::sym("phi", n_phi);

        // Value of potential at false 
        DM v_true = potential(true_vac);
        
        // Control intervals and spacing (evenly spaced for now)
        for (int i = 0; i < N; ++i) {
            double t = -1.0 + 2.0*i / N;
            double h = 2.0/N;
            t_k.push_back(t);
            h_k.push_back(h);
        }

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
        SXVector Phi, U, h_par, par;
        std::vector<double> lbPhi, ubPhi, lbU, ubU;

        // Limits for unbounded variables
        std::vector<double> ubinf(n_phi, inf);
        std::vector<double> lbinf(n_phi, -inf);

        /**** Initialise parameter variables ****/
        SXVector gamma_par, gammadot_par;
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

        // Zero vector for constraint bounds
        std::vector<double> zeroes(n_phi, 0);

        // Derivative at origin fixed to zero
        SX U_0_0 = SX::sym("U_0_0", n_phi);
        U.push_back(U_0_0);
        append_d(lbU, zeroes);
        append_d(ubU, zeroes);

        for (int k = 1; k < N; ++k) {
            SX Uk = SX::sym(varname("U", {k}), n_phi);
            U.push_back(Uk);
            append_d(lbU, lbinf);
            append_d(ubU, ubinf);
        }

        // Need to avoid Tr(1) singularity
        SX U_N_0 = SX::sym("U_N_0", n_phi);
        U.push_back(U_N_0);
        append_d(lbU, lbinf);
        append_d(ubU, ubinf);

        /**** Initialise state variables ****/

        // Free endpoint states
        std::vector<SX> endpoints;
        for (int k = 0; k < N; ++k) {
            SX phi_k_0 = SX::sym(varname("phi", {k, 0}), n_phi);
            endpoints.push_back(phi_k_0);
            Phi.push_back(phi_k_0);
            append_d(lbPhi, lbinf);
            append_d(ubPhi, ubinf);
        }

        // Final state, fixed to the false vacuum
        SX phi_N_0 = SX::sym("phi_N_0", n_phi);
        endpoints.push_back(phi_N_0);
        Phi.push_back(phi_N_0);
        append_d(lbPhi, false_vac.get_elements());
        append_d(ubPhi, false_vac.get_elements());

        // Build finite elements (including left endpoints)
        std::vector<SXVector> element_states;
        SXVector element_plot;
        for (int k = 0; k < N; ++k) {
            std::vector<SX> e_states;
            e_states.push_back(endpoints[k]);
            for (int j = 1; j <= d; ++j) {
                SX phi_k_j = SX::sym(varname("phi", {k, j}), n_phi);
                e_states.push_back(phi_k_j);
                Phi.push_back(phi_k_j);
                append_d(lbPhi, lbinf);
                append_d(ubPhi, ubinf);
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
            phidot_cons.push_back(h_elem*gammadot(j)*control_int[j - 1] - phidot_approx);
        }
        
        SXVector phidot_inputs;
        phidot_inputs.insert(phidot_inputs.end(), element.begin(), element.end());
        phidot_inputs.push_back(control_start);
        phidot_inputs.push_back(control_end);
        phidot_inputs.push_back(h_elem);
        phidot_inputs.push_back(gammadot);
        
        Function Phidot_cons = Function("Phidot_cons", phidot_inputs, phidot_cons);

        // Value of kinetic objective on an element
        SX T_k = 0;

        for (int j = 1; j <= d; ++j) {
            T_k = T_k + 0.5*S_n*h_elem*B[j]*pow(gamma(j), n_dims - 1)
                *gammadot(j)*dot(control_int[j - 1], control_int[j - 1]);
        }
        
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
            V_k = V_k + S_n*h_elem*B[j]*pow(gamma(j), n_dims - 1)*gammadot(j)*potential(element[j])[0];
        }

        Function V_cons_k = Function("V_cons_k", quadrature_inputs, {V_k});

        /**** Implement the objective and constraints ****/

        SXVector g = {}; // All constraints
        std::vector<double> lbg = {}; // Lower bounds for constraints
        std::vector<double> ubg = {}; // Upper bounds for constraints

        // Zero vector for collocation bounds
        std::vector<double> zeroes_col(d*n_phi, 0);

        // Continuity equations
        for (int k = 0; k < N; ++k) {
            g.push_back(Phi_end(element_states[k])[0] - endpoints[k + 1]);
            append_d(lbg, zeroes);
            append_d(ubg, zeroes);
        }

        // Build quadratures
        SX T = 0; // Objective function
        SX V = 0; // Integral constraint

        for (int k = 0; k < N; ++k) {
            SXVector quadrature_inputs_ = SXVector(element_states[k]);
            quadrature_inputs_.push_back(U[k]);
            quadrature_inputs_.push_back(U[k + 1]);
            quadrature_inputs_.push_back(h_par[k]);
            quadrature_inputs_.push_back(gamma_par[k]);
            quadrature_inputs_.push_back(gammadot_par[k]);
            T += T_obj_k(quadrature_inputs_)[0];
            V += V_cons_k(quadrature_inputs_)[0];
        }

        // Collocation equations and objective function
        for (int k = 0; k < N; ++k) {
            SXVector phidot_inputs_ = SXVector(element_states[k]);
            phidot_inputs_.push_back(U[k]);
            phidot_inputs_.push_back(U[k + 1]);
            phidot_inputs_.push_back(h_par[k]);
            phidot_inputs_.push_back(gammadot_par[k]);
            g.push_back(SX::vertcat(Phidot_cons(phidot_inputs_)));
            append_d(lbg, zeroes_col);
            append_d(ubg, zeroes_col);
        }

        /**** Calculate the ansatz ****/
        Function fV = Function("fV", {vertcat(Phi), vertcat(par)}, {V}, {"Phi", "par"}, {"V"});
        Ansatz a = get_ansatz(fV, par0, true_vac, false_vac);

        /**** Concatenate decision variables ****/

        SXVector w = {}; // All decision variables
        std::vector<double> w0 = {}; // Initial values for decision variables
        std::vector<double> lbw = {}; // Lower bounds for decision variables
        std::vector<double> ubw = {}; // Upper bounds for decision variables      
    
        // State variables
        w.insert(w.end(), Phi.begin(), Phi.end());
        w0.insert(w0.end(), a.Phi0.begin(), a.Phi0.end());
        lbw.insert(lbw.end(), lbPhi.begin(), lbPhi.end());
        ubw.insert(ubw.end(), ubPhi.begin(), ubPhi.end());

        // Control variables
        w.insert(w.end(), U.begin(), U.end());
        w0.insert(w0.end(), a.U0.begin(), a.U0.end());
        lbw.insert(lbw.end(), lbU.begin(), lbU.end());
        ubw.insert(ubw.end(), ubU.begin(), ubU.end());

        // Evaluate quadratures on ansatz
        Function fT = Function("fT", {vertcat(U), vertcat(par)}, {T}, {"U", "par"}, {"T"});
        DMDict argT;
        argT["U"] = a.U0;
        argT["par"] = par0;
        double T0 = fT(argT).at("T").get_elements()[0];

        DMDict argV;
        argV["Phi"] = a.Phi0;
        argV["par"] = par0;
        double V0 = fV(argV).at("V").get_elements()[0];

        std::cout << std::setprecision(20);
        std::cout << "V(ansatz) = " << V0 << std::endl;
        std::cout << "T(ansatz) = " << T0 << std::endl;

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
        nlp_opt["ipopt.tol"] = 1e-3;
        nlp_opt["ipopt.constr_viol_tol"] = 1e-3;

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

        // Evaluate the objective & constraint on the result
        Function T_ret = Function("T_ret", {W, Par}, {T}, {"W", "Par"}, {"T"});
        Function V_ret = Function("V_ret", {W, Par}, {V}, {"W", "Par"}, {"V"});
        DMDict T_ret_arg;
        T_ret_arg["W"] = res["x"];
        T_ret_arg["Par"] = par0;
        double Tret = T_ret(T_ret_arg).at("T").get_elements()[0];
        double Vret = V_ret(T_ret_arg).at("V").get_elements()[0];

        std::cout << "V(result) = " << Vret << std::endl;
        std::cout << "T(result) = " << Tret << std::endl;
        
        // Calculate the action
        double action = std::pow(((2.0 - d)/d)*(Tret/Vret), 0.5*d)*((2.0*Vret)/(2.0 - d));

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

        Eigen::VectorXd radii(N*(d + 1));
        int c = 0;
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j <= d; ++j) {
                radii(c) = Tr(t_kj(k, j));
                c++;
            }
        }
        
        auto t_extract_end = high_resolution_clock::now();
        auto extract_duration = duration_cast<microseconds>(t_extract_end - t_extract_start).count() * 1e-6;
        std::cout << "CasadiMaupertuisSolver - extracting / formatting took " << extract_duration << " sec" << std::endl;

        std::cout << W.size() << std::endl;

        return BouncePath(radii, elementmx, action);
    }
};

}

#endif