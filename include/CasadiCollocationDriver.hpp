#ifndef BUBBLETESTER_CASADICOLLOCATIONDRIVER_HPP_INCLUDED
#define BUBBLETESTER_CASADICOLLOCATIONDRIVER_HPP_INCLUDED

#include <exception>
#include <cmath>
#include "CasadiCommon.hpp"
#include "GenericBounceSolver.hpp"
#include "GenericPotential.hpp"
#include "CasadiPotential.hpp"
#include "BouncePath.hpp"

namespace BubbleTester {

class CasadiCollocationSolver : public GenericBounceSolver {
public:
    CasadiCollocationSolver(int n_spatial_dimensions_, int n_nodes_) : 
        n_spatial_dimensions(n_spatial_dimensions_), n_nodes(n_nodes_) {
        if (!(0 < n_nodes <= 50)) {
            
            throw std::invalid_argument("n_nodes must be between 1 and 50.");
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
        return "CasadiCollocation";
    }

    void set_verbose(bool verbose) override {
        // Do nothing for now
    }


private:
    int n_spatial_dimensions;
    int n_nodes;

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
        return true_vac + 0.5*(false_vac - true_vac)*(1 + tanh((rho - r0) / sigma));
    }

    //! Derivative of ansatz in semi-infinite coordinates
    casadi::DM ansatz_dot(double rho, casadi::DM true_vac, casadi::DM false_vac,
        double r0, double sigma) const {
        return (false_vac - true_vac) / (2*sigma*std::pow(cosh((rho - r0) / sigma), 2));
    }

    BouncePath _solve(const Eigen::VectorXd& true_vacuum, 
                      const Eigen::VectorXd& false_vacuum,
                      casadi::Function potential) const {
        using namespace casadi; 

        DM true_vac = eigen_to_dm(true_vacuum);
        DM false_vac = eigen_to_dm(false_vacuum);
        int n_phi = false_vac.size1();
        double d = get_n_spatial_dimensions();

        std::vector<double> collocation_points = radau_points[n_nodes - 1];
        std::vector<double> collocation_weights = radau_weights[n_nodes - 1];

        // Append the endpoint (1) to the collocation points
        // NB the weights are only used for integration, where the final
        // point is ignored.
        collocation_points.push_back(1.0);

        // TEMP - hard coded ansatz parameters
        double r0 = 5;
        double sigma = 3;
        
        // Kink ansatz
        auto ansatz_tau = [this, true_vac, false_vac, r0, sigma](double tau) {
            return ansatz(Tr(tau), true_vac, false_vac, r0, sigma).get_elements();
        };

        // Kink ansatz derivative
        auto dansatz_dtau = [this, true_vac, false_vac, r0, sigma](double tau) {
            // return (Tr_dot(tau)*ansatz_dot(Tr(tau),
            //     true_vac, false_vac, r0, sigma)).get_elements();
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
        std::vector<Polynomial> P;
        std::vector<Polynomial> Pder;

        for (int j = 0; j <= n_nodes; ++j) {
            Polynomial p = 1;
            for (int r = 0; r <= n_nodes; ++r) {
                if (r != j) {
                p *= Polynomial(-collocation_points[r],1)
                    / (collocation_points[j] - collocation_points[r]);
                }
            }
            P.push_back(p);
            Pder.push_back(p.derivative());
        }
        
        // D is (N*(N + 1)); here each entry in D represents a row (N entries).
        DMVector D;
        
        for (int r = 0; r < n_nodes; ++r) {
            std::vector<double> Drow;
            for (int c = 0; c <= n_nodes; ++c) {
                Drow.push_back(Pder[c](collocation_points[r]));
            }
            D.push_back(DM(Drow));
        }
        
        /**** Build the constraint functional ****/
        SX V = 0;
        for (int i = 0; i < n_nodes; ++i) {
            V += \
                collocation_weights[i]*
                std::pow(collocation_points[i], d - 1)*
                T[i]*potential({Phi[i]})[0];
        }

        // Evaluate the constraint functional on the ansatz to fix V0
        Function fV = Function("fV", {vertcat(Phi)}, {V});
        double V0 = fV(DM(Phi0))[0].get_elements()[0];

        /**** Build the cost functional ****/
        SX J = 0;
        for (int i = 0; i < n_nodes; ++i) {
            J += \
                collocation_weights[i]*
                std::pow(collocation_points[i], d - 1)*
                T[i]*norm_2(U[i]);
        }

        // TEMP - evaluate cost functional on ansatz cos why not
        Function fT = Function("fT", {vertcat(U)}, {J});
        double T0 = fT(DM(U0))[0].get_elements()[0];

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

        // Dynamic constraints
        for (int i = 0; i < n_nodes; ++i) {
            SX dphi_i = 0;
            for (int j = 0; j <= n_nodes; ++j) {
                // double D_ij = D[i].get_elements()[j];
                double D_ij = Pder[j](collocation_points[i]);
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
        DMDict arg = {{"x0", w0}, {"lbx", lbw}, {"ubx", ubw}, {"lbg", lbg}, {"ubg", ubg}};
        DMDict res = solver(arg);

        // TEMP
        // Check the dynamics constraints on the ansatz (should be all ~0)
        Function dynF = Function("dynF", {vertcat(Phi), vertcat(U)}, {G});
        DMVector dynArg = {DM(Phi0), DM(U0)};
        DMVector dynVal = dynF(dynArg);
        std::cout << "dynF: " << std::endl << dynVal << std::endl;

        // Check the derivative estimates 
        Function derF = Function("derF", {vertcat(Phi)}, {vertcat(dphi)});
        DMVector derArg = {DM(Phi0)};
        DMVector derVal = derF(derArg);
        std::cout << "derF: " << std::endl << derVal << std::endl;
        
        std::cout << std::setprecision(20);
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
        std::cout << "U0:" << U0 << std::endl;

        // std::cout << "T: " << T << std::endl;
        // std::cout << "w: " << w << std::endl;
        // std::cout << "W: " << W << std::endl;
        // std::cout << "g: " << g << std::endl;
        // std::cout << "G: " << G << std::endl;

        // std::cout << "Differentiation matrix:" << std::endl;
        // for (int r = 0; r < n_nodes; ++r) {
        //     for (int c = 0; c <= n_nodes; ++c) {
        //         std::cout << D[r].get_elements()[c] << '\t';
        //     }
        //     std::cout << std::endl;
        // }    

        // std::cout << "P: " << std::endl << P << std::endl;
        // std::cout << "Pder: " << std::endl << Pder << std::endl;

        // Return dummy bounce path for now
        return BouncePath();

    }
};

}

#endif