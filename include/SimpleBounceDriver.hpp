#ifndef BUBBLETESTER_SIMPLEBOUNCEDRIVER_HPP_INCLUDED
#define BUBBLETESTER_SIMPLEBOUNCEDRIVER_HPP_INCLUDED

#include "GenericBounceSolver.hpp"
#include "GenericPotential.hpp"
#include "BouncePath.hpp"
#include "SimpleBounce.hpp"

namespace BubbleTester {

//////// Wrapper class for simplebounce::GenericModel

class SimpleBouncePotential : public simplebounce::GenericModel {
public:
    SimpleBouncePotential(const GenericPotential& potential_) :
        potential(potential_) {
        const int nPhi = potential.get_number_of_fields();
        setNphi(nPhi);
    }

    double vpot (const double* phi) const {
        // *chuckles* I'm in danger...
        Eigen::VectorXd phi_vec(nphi());
        for (int i = 0; i < nphi(); ++i) {
            phi_vec(i) = phi[i];
        }

        return potential(phi_vec);
    }

    void calcDvdphi(const double *phi, double *dvdphi) const {
        Eigen::VectorXd phi_vec(nphi());
        for (int i = 0; i < nphi(); ++i) {
            phi_vec(i) = phi[i];
        }
        
        for (int j = 0; j < nphi(); ++j) {
            dvdphi[j] = potential.partial(phi_vec, j);
        }
    }

private:
    const GenericPotential& potential;
};

//////// Wrapper class for simplebounce::BounceCalculator

class SimpleBounceSolver : public GenericBounceSolver {
public:

    SimpleBounceSolver(double rmax_, int grid_) {
        rmax = rmax_;
        grid = grid_;
    }

    BouncePath solve(
        const Eigen::VectorXd& true_vacuum,
        const Eigen::VectorXd& false_vacuum,
        const GenericPotential& potential) const override {

            int n_fields = potential.get_number_of_fields();

            simplebounce::BounceCalculator bounce;
            bounce.setRmax(rmax);
            bounce.setDimension(3);
            bounce.setN(grid);
            bounce.setNphi(n_fields);
            if (verbose) bounce.verboseOn();

            // needs delete
            simplebounce::GenericModel *model = new SimpleBouncePotential(potential);

            // don't need delete
            const double *tv = true_vacuum.data();
            const double *fv = false_vacuum.data();

            bounce.setModel(model);
            bounce.setVacuum(tv, fv);
            bounce.solve();

            // Extract results
            Eigen::VectorXd radii(grid);
            for (int i = 0; i < grid; ++i) {
                radii(i) = bounce.r(i);
            }
            
            Eigen::MatrixXd fields(grid, n_fields);
            for (int i = 0; i < grid; ++i) {
                for (int j = 0; j < n_fields; ++j) {
                    fields(i, j) = bounce.phi(i, j);
                }
            }

            double action = bounce.action();

            delete model;
            return BouncePath(radii, fields, action);
        }
    
    void set_verbose(bool verbose_) override {
        verbose = verbose_;
    }

    std::string name() override {
        return "SimpleBounce";
    }


private:
    bool verbose = false;
    double rmax;
    int grid;
    
};



};

#endif