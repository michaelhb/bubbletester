#ifndef BUBBLETESTER_CASADICOLLOCATIONDRIVER_HPP_INCLUDED
#define BUBBLETESTER_CASADICOLLOCATIONDRIVER_HPP_INCLUDED

#include <exception>
#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>

#include "CasadiCommon.hpp"
#include "GenericBounceSolver.hpp"
#include "GenericPotential.hpp"
#include "CasadiPotential.hpp"
#include "BouncePath.hpp"
#include <boost/math/tools/polynomial.hpp>

namespace BubbleTester {

class CasadiCollocationSolver : public GenericBounceSolver {
public:
    CasadiCollocationSolver(int n_spatial_dimensions_, int n_nodes_) : 
        n_spatial_dimensions(n_spatial_dimensions_), n_nodes(n_nodes_) {
        if (!(0 < n_nodes <= 50)) {
            throw std::invalid_argument("n_nodes must be between 1 and 50.");
        }

        // Read in collocation frame from files (fix hardcoded paths later)
        std::string prefix = "/home/michael/SciCodes/bubbletester/wls/frames/";
        std::ostringstream fn_points, fn_weights, fn_d;
        
        fn_points << prefix << "points_" << n_nodes << ".tsv";
        fn_weights << prefix << "weights_" << n_nodes << ".tsv";
        fn_d << prefix << "d_" << n_nodes << ".tsv";

        double dt;
        std::ifstream ifs;

        ifs = std::ifstream(fn_points.str());
        while (ifs >> dt) {
            collocation_points.push_back(dt);
        }

        ifs = std::ifstream(fn_weights.str());
        while (ifs >> dt) {
            collocation_weights.push_back(dt);
        }

        ifs = std::ifstream(fn_d.str());
        for (int r = 0; r <= n_nodes; ++r) {
            std::vector<double> Drow;
            for (int c = 0; c <= n_nodes; ++c) {
                ifs >> dt;
                Drow.push_back(dt);
            }
            D.push_back(Drow);
        }

        // Append the endpoint (1) to the collocation points
        // NB the weights are only used for integration, where the final
        // point is ignored.
        collocation_points.push_back(1.0);
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
            std::cout << "CasadiCollocationSolver: this is a CasadiPotential" << std::endl;
            Function potential = c_potential.get_function(); 
            return _solve(true_vacuum, false_vacuum, potential);
        }
        catch (const std::bad_cast) {
            // Case 2: we are not working with a CasadiPotential,
            // so we want to wrap it in a Callback and use finite 
            // differences to calculate derivatives.
            std::cout << "CasadiCollocationSolver: this is not a CasadiPotential" << std::endl;
            CasadiPotentialCallback cb(g_potential);
            Function potential = cb;
            return _solve(true_vacuum, false_vacuum, potential);
        }
    }

    int get_n_spatial_dimensions() const override {
        return n_spatial_dimensions;
    }

    std::string name() override {
        return "CasadiCollocation";
    }

    void set_verbose(bool verbose) override {
        // Do nothing for now
    }


private:
    int n_spatial_dimensions;
    int n_nodes;
    std::vector<double> collocation_points; 
    std::vector<double> collocation_weights;
    std::vector<std::vector<double>> D;

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

    BouncePath _solve(const Eigen::VectorXd& true_vacuum, 
                      const Eigen::VectorXd& false_vacuum,
                      casadi::Function potential) const {
        using namespace casadi; 

        DM true_vac = eigen_to_dm(true_vacuum);
        DM false_vac = eigen_to_dm(false_vacuum);
        int n_phi = false_vac.size1();
        double d = get_n_spatial_dimensions();

        // TEMP - hard coded ansatz parameters (r0 = 2., sigma=.5 good for delta = 0.4)
        double r0 = 2.0;
        double sigma = 0.5;

        // TEMP - hard coded volume factors
        double S_n;
        if (d == 3) {
            S_n = 4*pi;
        }
        else if (d == 4) {
            S_n = 0.5*pi*pi;
        }
        else {
            throw std::invalid_argument("Only d = 3 and d = 4 are currently supported.");
        }
        
        // Kink ansatz
        auto ansatz_tau = [this, true_vac, false_vac, r0, sigma](double tau) {
            return ansatz(Tr(tau), true_vac, false_vac, r0, sigma).get_elements();
        };

        // Kink ansatz derivative
        auto dansatz_dtau = [this, true_vac, false_vac, r0, sigma](double tau) {
            return (ansatz_dot(Tr(tau),
                true_vac, false_vac, r0, sigma)).get_elements();
        };

        // Take derivatives of the monotonic transformation at the collocation point,
        // excluding the endpoint (these are used to make the quadrature invariant
        // under the transform)
        std::vector<double> T;
        for (int i = 0; i < n_nodes; ++i) {
            T.push_back(Tr_dot(collocation_points[i]));
        }

        // Set up the state ((N + 1)*n) and control (N*n) matrices,
        // and the initial (ansatz) values
        SXVector Phi, U;
        std::vector<double> Phi0, U0;

        for (int i = 0; i <= n_nodes; ++i) {
            Phi.push_back(SX::sym(varname("phi",{i}), n_phi));
        }
        for (int i = 0; i < n_nodes; ++i) {
            U.push_back(SX::sym(varname("u",{i}), n_phi));
            append_d(Phi0, ansatz_tau(collocation_points[i]));
            append_d(U0, dansatz_dtau(collocation_points[i]));
        }
        for (int i = 0; i < n_phi; ++i) { // endpoint for Phi only
            Phi0.push_back(false_vacuum(i));
        }

        /**** Set up the boundary conditions ****/
        std::vector<double> lbPhi(n_phi*(n_nodes + 1), -inf);
        std::vector<double> ubPhi(n_phi*(n_nodes + 1), inf);
        std::vector<double> lbU(n_phi*n_nodes, -inf);
        std::vector<double> ubU(n_phi*n_nodes, inf);

        // Derivative at origin is zero
        for (int i = 0; i < n_phi; ++i) {
            lbU[i] = 0;
            ubU[i] = 0;
        }        

        // State goes to false vacuum at infinity 
        for (int i = 0; i < n_phi; ++i) {
            lbPhi[n_phi*(n_nodes) + i] = false_vacuum(i);
            ubPhi[n_phi*(n_nodes) + i] = false_vacuum(i);
        }

        /**** Construct the Gauss pseudospectral differentiation matrix (D). ****/

        // Build the Lagrange basis, and take derivatives of its elements. 
        GiNaC::symbol t("r");
        std::vector<GiNaC::ex> P = lagrange_basis(t, collocation_points);
        std::vector<GiNaC::ex> Pder;
        for (int i = 0; i < P.size(); ++i) {
            Pder.push_back(P[i].diff(t, 1));
        }
        
        /**** Build the constraint functional ****/
        SX V = 0;
        for (int i = 0; i < n_nodes; ++i) {
            V += \
                S_n*collocation_weights[i]*
                std::pow(Tr(collocation_points[i]), d - 1)*
                Tr_dot(collocation_points[i])*potential({Phi[i]})[0];
        }

        // Evaluate the constraint functional on the ansatz to fix V0
        Function fV = Function("fV", {vertcat(Phi)}, {V});
        double V0 = fV(DM(Phi0))[0].get_elements()[0];

        /**** Build the cost functional ****/
        SX J = 0;
        for (int i = 0; i < n_nodes; ++i) {
            J += \
                0.5*S_n*collocation_weights[i]*
                std::pow(Tr(collocation_points[i]), d - 1)*
                Tr_dot(collocation_points[i])*dot(U[i],U[i]);
        }

        // Evaluate cost functional on ansatz 
        Function fT = Function("fT", {vertcat(U)}, {J});
        double T0 = fT(DM(U0))[0].get_elements()[0];

        // Evaluate gradient of cost functional on ansatz
        SX gradJ = gradient(J, vertcat(U));
        Function gradT = Function("gradT", {vertcat(U)}, {gradJ});
        DMVector gradT0 = gradT(DM(U0));

        /**** Concatenate all decision variables and constraints ****/
        
        SXVector w = {}; // All decision variables
        std::vector<double> w0 = {}; // Initial values for decision variables
        std::vector<double> lbw = {}; // Lower bounds for decision variables
        std::vector<double> ubw = {}; // Upper bounds for decision variables      
        
        SXVector g = {}; // All constraints
        std::vector<double> lbg = {}; // Lower bounds for constraints
        std::vector<double> ubg = {}; // Upper bounds for constraints

        // State variables
        w.insert(w.end(), Phi.begin(), Phi.end());
        w0.insert(w0.end(), Phi0.begin(), Phi0.end());
        lbw.insert(lbw.end(), lbPhi.begin(), lbPhi.end());
        ubw.insert(ubw.end(), ubPhi.begin(), ubPhi.end());

        // Control variables
        w.insert(w.end(), U.begin(), U.end());
        w0.insert(w0.end(), U0.begin(), U0.end());
        lbw.insert(lbw.end(), lbU.begin(), lbU.end());
        ubw.insert(ubw.end(), ubU.begin(), ubU.end());

        // Zero vector for constraint bounds
        std::vector<double> zeroes(n_phi, 0);

        // TEMP - save the dphi_i estimates
        SXVector dphi;

        // Potential constraint
        g.push_back(V0 - V);
        // g.push_back(V + .5);
        lbg.push_back(0);
        ubg.push_back(0);
        
        // Dynamic constraints
        for (int i = 0; i < n_nodes; ++i) {
            SX dphi_i = 0;
            for (int j = 0; j <= n_nodes; ++j) {
                double D_ij = D[i][j];
                dphi_i += D_ij*Phi[j];
            }
            dphi.push_back(dphi_i);
            g.push_back(dphi_i - T[i]*U[i]);
            lbg.insert(lbg.end(), zeroes.begin(), zeroes.end());
            ubg.insert(ubg.end(), zeroes.begin(), zeroes.end());
        }
        
        // Collect states & constraints into single vectors
        SX W = vertcat(w);
        SX G = vertcat(g);

        // Create the solver
        SXDict nlp = {{"f", J}, {"x", W}, {"g", G}};
        Dict nlp_opt = Dict();
        Function solver = nlpsol("nlpsol", "ipopt", nlp, nlp_opt);

        // Run the optimiser
        Function Phi_ret = Function("Phi_ret", {W}, {SX::horzcat(Phi)});
        Function U_ret = Function("U_ret", {W}, {SX::horzcat(U)});
        
        DMDict arg = {{"x0", w0}, {"lbx", lbw}, {"ubx", ubw}, {"lbg", lbg}, {"ubg", ubg}};
        DMDict res = solver(arg);

        // Extract & format path / profiles / multiplier
        DMVector rPhi = Phi_ret(res["x"]);
        DMVector rU = U_ret(res["x"]);

        Eigen::MatrixXd profiles(n_nodes, n_phi); // Exclude endpoint for now
        for (int c = 0; c < n_phi; c++) {
            std::vector<double> col = vertsplit(rPhi[0])[c].get_elements();
            for (int r = 0; r < n_nodes; ++r) { 
                profiles(r, c) = col[r];
            }
        }

        // Evaluate cost & constraint on results
        Function T_ret = Function("T_ret", {W}, {J});
        double rT0 = T_ret(res["x"])[0].get_elements()[0];
        Function V_ret = Function("V_ret", {W}, {V});
        double rV0 = V_ret(res["x"])[0].get_elements()[0];

        // Calculate the action
        double action = std::pow(((2.0 - d)/d)*(rT0/rV0), 0.5*d)*((2.0*rV0)/(2.0 - d));

        // Do the alternative calculation as a check
        double lam_star2 = ((2.0 - d)/d)*(rT0/rV0);
        double action2 = std::pow(((2.0 - d)/d)*(rT0/rV0), 0.5*d - 1)*((2.0*rT0)/d);
        std::cout << ">>> action 1: " << action << std::endl;
        std::cout << ">>> action 2: " << action2 << std::endl;
        std::cout << ">>> inferred lambda_star: " << lam_star2 << std::endl;
        std::cout << ">>> IPOPT lambda_star: " << res["lam_g"].get_elements()[0] << std::endl;

        // Print results
        // std::cout << "==== Phi_ret ====" << std::endl;
        // std::cout << rPhi << std::endl;
        // std::cout << "==== U_ret ====" << std::endl;
        // std::cout << rU << std::endl;
        // std::cout << "----" << std::endl;
        std::cout << std::setprecision(20) << std::endl;
        std::cout << "T(result) = " << rT0 << std::endl;
        std::cout << "V(result) = " << rV0 << std::endl;
        std::cout << "----" << std::endl;
        // // TEMP
        // // Check the dynamics constraints on the ansatz (should be all ~0)
        // Function dynF = Function("dynF", {vertcat(Phi), vertcat(U)}, {G});
        // DMVector dynArg1 = {DM(Phi0), DM(U0)};
        // DMVector dynVal1 = dynF(dynArg1);
        // std::cout << "dynF (ansatz): " << std::endl << dynVal1 << std::endl;

        // Check the derivative estimates 
        // Function derF = Function("derF", {vertcat(Phi)}, {vertcat(dphi)});
        // DMVector derArg = {DM(Phi0)};
        // DMVector derVal = derF(derArg);
        // std::cout << "derF: " << std::endl << derVal << std::endl;
        
        // std::cout << std::setprecision(20);
        // std::cout << "Collocation points:" << std::endl;
        // for (int i = 0; i <= n_nodes; ++i) {
        //     std::cout << collocation_points[i] << std::endl;
        // }

        // std::cout << "Points on semi infinite domain (excl. infty): " << std::endl;
        // for (int i = 0; i < n_nodes; ++i) {
        //     std::cout << Tr(collocation_points[i]) << std::endl;
        // }

        std::cout << "V(ansatz) = " << V0 << std::endl;
        std::cout << "T(ansatz) = " << T0 << std::endl;
        // std::cout << "gradT(ansatz):" << std::endl << gradT0 << std::endl;
        // std::cout << "State matrix:" << std::endl << Phi << std::endl;
        // std::cout << "Control matrix:" << std::endl << U << std::endl;
        // std::cout << "Differentiation matrix:" << std::endl << D << std::endl;
        // std::cout << "Constraint functional:" << std::endl << V << std::endl;
        // std::cout << "Cost functional:" << std::endl << J << std::endl;

        // std::cout << "ubPhi: " << ubPhi << std::endl;
        // std::cout << "lbPhi: " << lbPhi << std::endl;
        // std::cout << "ubU: " << ubU << std::endl;
        // std::cout << "lbU: " << lbU << std::endl;
        // std::cout << "Phi0:" << Phi0 << std::endl;
        // std::cout << "U0:" << U0 << std::endl;

        // std::cout << "T: " << T << std::endl;
        // std::cout << "w: " << w << std::endl;
        // std::cout << "W: " << W << std::endl;
        // std::cout << "w0:" << w0 << std::endl;
        // std::cout << "g: " << g << std::endl;
        // std::cout << "G: " << G << std::endl;
        Eigen::VectorXd radii(collocation_points.size() - 1);
        
        for (int i = 0; i < collocation_points.size() - 1; ++i) {
            radii(i) = Tr(collocation_points[i]);
        }
        return BouncePath(radii, profiles, action);
    }

    
};

}

#endif