#ifndef BUBBLETESTER_CASADICOLLOCATIONDRIVER2_HPP_INCLUDED
#define BUBBLETESTER_CASADICOLLOCATIONDRIVER2_HPP_INCLUDED

#include <memory>
#include <chrono>
#include <exception>
#include <math.h> 
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

struct NLP {
    casadi::Function nlp;
    
    // Separate T/V for ansatz / return is 
    // ugly and should be done better 
    casadi::Function T_a;
    casadi::Function T_ret;
    casadi::Function V_a;
    casadi::Function V_ret;
    casadi::Function Phi_ret;
};

class CasadiCollocationSolver2 : public GenericBounceSolver {
public:
    CasadiCollocationSolver2(int n_phi_, int n_spatial_dimensions_, int N_) : n_phi(n_phi_), n_dims(n_spatial_dimensions_), N(N_), d(3) {
        using namespace casadi;
        tau_root = collocation_points(d, "legendre");
        tau_root.insert(tau_root.begin(), 0.);

        if (n_dims == 3) {
            S_n = 4*pi;
        }
        else if (n_dims == 4) {
            S_n = 0.5*pi*pi;
        }
        else {
            throw std::invalid_argument("Only d = 3 and d = 4 are currently supported.");
        }

        // Control intervals and spacing (evenly spaced for now)
        for (int i = 0; i < N; ++i) {
            double t = -1.0 + 2.0*i / N;
            double h = 2.0/N;
            t_k.push_back(t);
            h_k.push_back(h);
        }

        // Initialise polynomial basis and collocation / integration coefficients
        // TODO - this should be done offline
        D = std::vector<double>(d + 1, 0);
        B = std::vector<double>(d + 1, 0);
        P = std::vector<Polynomial>(d + 1);
        C = std::vector<std::vector<double> >(d+1,std::vector<double>(d+1, 0));

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
    int n_phi; // Number of field dimensions
    int n_dims; // Number of spatial dimensions
    double S_n; // Surface area of (d-1)-sphere
    int N; // Number of finite elements
    int d; // Degree of interpolating polynomials
    double grid_scale = 15.0;

    std::vector<double> t_k; // Element start times
    std::vector<double> h_k; // Element widths

    // Coefficients of the collocation equation
    std::vector<std::vector<double> > C;

    // Coefficients of the continuity equation
    std::vector<double> D;

    // Coefficients for Gaussian quadrature 
    std::vector<double> B;

    // The polynomial basis
    std::vector<casadi::Polynomial> P;

    // Vector of collocation points on [0,1)
    std::vector<double> tau_root;

    // Time at element k, collocation point j
    double t_kj(int k, int j) const {
        return t_k[k] + h_k[k]*tau_root[j];
    }

    //! Transformation from compact to semi infinite domain
    double Tr(double tau) const {
        return grid_scale*log(2.0/(1.0 - tau));
    }

    //! Derivative of monotonic transformation
    double Tr_dot(double tau) const {
        return grid_scale / (1.0 - tau);
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
        std::vector<double> grid_pars, casadi::DM true_vac, casadi::DM false_vac) const {

        Ansatz a;
        std::vector<double> Phi0, U0;
        Phi0.reserve((N*(n_dims + 1) + 1));
        U0.reserve(N + 1);
        casadi::DMDict argV;
        argV["par"] = grid_pars;

        double r0 = 0.5;
        double delta_r0 = 0.1;
        double r0_max = 10.;
        double targetV = -1;
        double tol = 1e-3;
        double sig_upper0 = 3.;
        double sig_upper = sig_upper0;
        double sig_lower = 0.;
        double sigma_min = 0.1;
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
            
            if (sigma < sigma_min) {
                std::cout << "Increasing radius by " << delta_r0 << std::endl;
                r0 += delta_r0;
                sig_upper = sig_upper0;
                sig_lower = 0;
                sigma = 0.5*(sig_lower + sig_upper);
                V_mid = 1; // Make sure we loop again

                if (r0 > r0_max) {
                    throw std::runtime_error("Exceeded maximum radius!");
                }
            }
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

    // Get vector of parametric input values
    std::vector<double> get_grid_pars() const {
        // This is a standin to allow for more flexibility later
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
        return par0;
    }

    // TODO memoize this, should only be called when potential changes
    // TODO make V0 a parameter, and push ansatz calculation to callers
    //   (then remove the vacuum parameters, they are only needed for V0)
    NLP get_nlp(casadi::Function potential) const {
        using namespace casadi;
        using namespace std::chrono;
        auto t_setup_start = high_resolution_clock::now();

        /**** Initialise parameter variables ****/
        SXVector h_par, gamma_par, gammadot_par; 
        SXVector grid_pars; // All grid parameter variables

        for (int i = 0; i < N; ++i) {
            SX h_par_ = SX::sym(varname("h", {i}));
            h_par.push_back(h_par_);
            grid_pars.push_back(h_par_);
        }
        for (int i = 0; i < N; ++i) {
            SX gamma_par_ = SX::sym(varname("gamma", {i}), d + 1);
            gamma_par.push_back(gamma_par_);
            grid_pars.push_back(gamma_par_);
        }
        for (int i = 0; i < N; ++i) {
            SX gammadot_par_ = SX::sym(varname("gammadot", {i}), d + 1);
            gammadot_par.push_back(gammadot_par_);
            grid_pars.push_back(gammadot_par_);
        }

        SX V0_par = SX::sym("V0");

        // Grid pars + V0
        SXVector pars = grid_pars;
        pars.push_back(V0_par);

        /**** Initialise control variables ****/        
        SXVector U;

        // Derivative at origin fixed to zero
        SX U_0_0 = SX::sym("U_0_0", n_phi);
        U.push_back(U_0_0);

        for (int k = 1; k <= N; ++k) {
            SX Uk = SX::sym(varname("U", {k}), n_phi);
            U.push_back(Uk);
        }

        /**** Initialise state variables ****/
        SXVector Phi;
        std::vector<SXVector> element_states; // States within an element
        SXVector element_plot; // Concatenated states within an element
        SXVector endpoints; // Endpoint states

        // Free endpoint states
        for (int k = 0; k < N; ++k) {
            SX phi_k_0 = SX::sym(varname("phi", {k, 0}), n_phi);
            endpoints.push_back(phi_k_0);
            Phi.push_back(phi_k_0);
        }

        // Final state, fixed to the false vacuum
        SX phi_N_0 = SX::sym("phi_N_0", n_phi);
        endpoints.push_back(phi_N_0);
        Phi.push_back(phi_N_0);

        // Build finite elements (including left endpoints)
        for (int k = 0; k < N; ++k) {
            std::vector<SX> e_states;
            e_states.push_back(endpoints[k]);
            for (int j = 1; j <= d; ++j) {
                SX phi_k_j = SX::sym(varname("phi", {k, j}), n_phi);
                e_states.push_back(phi_k_j);
                Phi.push_back(phi_k_j);
            }
            element_plot.push_back(SX::horzcat(e_states));
            element_states.push_back(e_states);
        }

        /**** Useful functions of the state and control variables in an element ****/

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

        // Value of potential constraint functional on an element
        SX V_k = 0;

        for (int j = 1; j <= d; ++j) {
            V_k = V_k + S_n*h_elem*B[j]*pow(gamma(j), n_dims - 1)*gammadot(j)*potential(element[j])[0];
        }

        // We define per-element functions for the quadratures, then use them to build
        // integrals over the whole domain.
        SXVector quadrature_inputs;
        quadrature_inputs.insert(quadrature_inputs.end(), element.begin(), element.end());
        quadrature_inputs.push_back(control_start);
        quadrature_inputs.push_back(control_end);
        quadrature_inputs.push_back(h_elem);
        quadrature_inputs.push_back(gamma);
        quadrature_inputs.push_back(gammadot);
        
        Function T_obj_k = Function("T_obj_k", quadrature_inputs, {T_k});
        Function V_cons_k = Function("V_cons_k", quadrature_inputs, {V_k});

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
        Function fV = Function("fV", {vertcat(Phi), vertcat(grid_pars)}, {V}, {"Phi", "par"}, {"V"});
        Function fT = Function("fT", {vertcat(U), vertcat(grid_pars)}, {T}, {"U", "par"}, {"T"});

        /**** Build constraints ****/ 
        SXVector g = {}; // All constraints

        // Continuity equations
        for (int k = 0; k < N; ++k) {
            g.push_back(Phi_end(element_states[k])[0] - endpoints[k + 1]);
        }

        // Collocation equations
        for (int k = 0; k < N; ++k) {
            SXVector phidot_inputs_ = SXVector(element_states[k]);
            phidot_inputs_.push_back(U[k]);
            phidot_inputs_.push_back(U[k + 1]);
            phidot_inputs_.push_back(h_par[k]);
            phidot_inputs_.push_back(gammadot_par[k]);
            g.push_back(SX::vertcat(Phidot_cons(phidot_inputs_)));
        }

        // Potential constraint
        g.push_back(V0_par - V);

        /**** Concatenate variables ****/
        SXVector w = {}; // All decision variables
        w.insert(w.end(), Phi.begin(), Phi.end());
        w.insert(w.end(), U.begin(), U.end());

        // Collect states and constraints into single vectors
        SX W = SX::vertcat(w);
        SX G = SX::vertcat(g);

        // Versions of T and V suitable for evaluating on results 
        SX elements_plot = SX::horzcat(element_plot);
        Function Phi_ret = Function("elements", {W}, {elements_plot});
        Function T_ret = Function("T_ret", {W, vertcat(grid_pars)}, {T}, {"W", "Par"}, {"T"});
        Function V_ret = Function("V_ret", {W, vertcat(grid_pars)}, {V}, {"W", "Par"}, {"V"});

        SXDict nlp_arg = {{"f", T}, {"x", W}, {"g", G}, {"p", vertcat(pars)}};
        Dict nlp_opt = Dict();
        nlp_opt["expand"] = false;
        nlp_opt["ipopt.tol"] = 1e-3;
        nlp_opt["ipopt.constr_viol_tol"] = 1e-3;

        Function solver = nlpsol("nlpsol", "ipopt", nlp_arg, nlp_opt);
        auto t_setup_end = high_resolution_clock::now();
        auto setup_duration = duration_cast<microseconds>(t_setup_end - t_setup_start).count() * 1e-6;
        std::cout << "CasadiMaupertuisSolver - setup took " << setup_duration << " sec" << std::endl;
        
        NLP nlp;
        nlp.nlp = solver;
        nlp.T_a = fT;
        nlp.T_ret = T_ret;
        nlp.V_a = fV;
        nlp.V_ret = V_ret;
        nlp.Phi_ret = Phi_ret;

        return nlp;
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

        // Get the grid parameters (h_k, gamma, gamma_dot)
        std::vector<double> grid_pars = get_grid_pars();

        // Initialise NLP and get ansatz solution
        NLP nlp = get_nlp(potential);
        Ansatz a = get_ansatz(nlp.V_a, grid_pars, true_vac, false_vac);

        // Evaluate T and V on the ansatz
        DMDict argT;
        argT["U"] = a.U0;
        argT["par"] = grid_pars;
        double T0 = nlp.T_a(argT).at("T").get_elements()[0];

        DMDict argV;
        argV["Phi"] = a.Phi0;
        argV["par"] = grid_pars; 
        double V0 = nlp.V_a(argV).at("V").get_elements()[0];

        std::cout << std::setprecision(20);
        std::cout << "V(ansatz) = " << V0 << std::endl;
        std::cout << "T(ansatz) = " << T0 << std::endl;
        
        /**** Bounds on control variables ****/
        // Need to find a less opaque way of ensuring that 
        // concatenated NLP inputs / bounds / start values 
        // are consistently ordered!

        // Limits for unbounded variables
        std::vector<double> ubinf(n_phi, inf);
        std::vector<double> lbinf(n_phi, -inf);

        // Zero vector for constraint bounds
        std::vector<double> zeroes(n_phi, 0);
        
        // Zero vector for collocation bounds
        std::vector<double> zeroes_col(d*n_phi, 0);
        std::vector<double> lbU, ubU;

        // Derivative at origin fixed to zero
        append_d(lbU, zeroes);
        append_d(ubU, zeroes);

        for (int k = 1; k <= N; ++k) {
            append_d(lbU, lbinf);
            append_d(ubU, ubinf);
        }

        /**** Bounds on state variables ****/
        std::vector<double> lbPhi, ubPhi;

        // Free endpoint states
        for (int k = 0; k < N; ++k) {
            append_d(lbPhi, lbinf);
            append_d(ubPhi, ubinf);
        }

        // Final state, fixed to the false vacuum
        append_d(lbPhi, false_vac.get_elements());
        append_d(ubPhi, false_vac.get_elements());

        // Free intermediate states
        for (int k = 0; k < N; ++k) {
            for (int j = 1; j <= d; ++j) {
                append_d(lbPhi, lbinf);
                append_d(ubPhi, ubinf);
            }
        }

        /**** Bounds on constraints ****/
        std::vector<double> lbg = {}; // Lower bounds for constraints
        std::vector<double> ubg = {}; // Upper bounds for constraints

        // Continuity equations
        for (int k = 0; k < N; ++k) {
            append_d(lbg, zeroes);
            append_d(ubg, zeroes);
        }

        // Collocation equations and objective function
        for (int k = 0; k < N; ++k) {
            append_d(lbg, zeroes_col);
            append_d(ubg, zeroes_col);
        }

        // Add potential constraint
        lbg.push_back(0);
        ubg.push_back(0);

        /**** Concatenate NLP inputs ****/

        std::vector<double> w0 = {}; // Initial values for decision variables
        w0.insert(w0.end(), a.Phi0.begin(), a.Phi0.end());
        w0.insert(w0.end(), a.U0.begin(), a.U0.end());
    
        /**** Concatenate decision variable bounds****/
        std::vector<double> lbw = {}; 
        std::vector<double> ubw = {};  

        lbw.insert(lbw.end(), lbPhi.begin(), lbPhi.end());
        ubw.insert(ubw.end(), ubPhi.begin(), ubPhi.end());
        lbw.insert(lbw.end(), lbU.begin(), lbU.end());
        ubw.insert(ubw.end(), ubU.begin(), ubU.end());

        /**** Initialise and solve the NLP ****/

        // Add V0 to parameters
        std::vector<double> pars(grid_pars);
        pars.push_back(V0);

        // Run the optimiser. This is the other bottleneck, so we time it too.
        DMDict arg = {{"x0", w0}, {"lbx", lbw}, {"ubx", ubw}, {"lbg", lbg}, {"ubg", ubg}, {"p", pars}};
        auto t_solve_start = high_resolution_clock::now();
        DMDict res = nlp.nlp(arg);
        auto t_solve_end = high_resolution_clock::now();
        auto solve_duration = duration_cast<microseconds>(t_solve_end - t_solve_start).count() * 1e-6;
        std::cout << "CasadiMaupertuisSolver - optimisation took " << solve_duration << " sec" << std::endl;

        // Evaluate the objective & constraint on the result
        DMDict ret_arg;
        ret_arg["W"] = res["x"];
        ret_arg["Par"] = grid_pars;
        double Tret = nlp.T_ret(ret_arg).at("T").get_elements()[0];
        double Vret = nlp.V_ret(ret_arg).at("V").get_elements()[0];

        std::cout << "V(result) = " << Vret << std::endl;
        std::cout << "T(result) = " << Tret << std::endl;
        
        // Calculate the action
        double action = std::pow(((2.0 - d)/d)*(Tret/Vret), 0.5*d)*((2.0*Vret)/(2.0 - d));

        // Return the result (interpolated using the Lagrange representation)

        Eigen::MatrixXd elementmx = 
            Eigen::Map<Eigen::MatrixXd>(
                nlp.Phi_ret(res["x"]).at(0).get_elements().data(), n_phi, N*(d + 1)).transpose();

        Eigen::VectorXd radii(N*(d + 1));
        int c = 0;
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j <= d; ++j) {
                radii(c) = Tr(t_kj(k, j));
                c++;
            }
        }

        return BouncePath(radii, elementmx, action);
    }
};

}

#endif