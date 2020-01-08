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
            phi_vec << phi[i];
        }

        return potential(phi_vec);
    }

    void calcDvdphi(const double *phi, double *dvdphi) const {
        Eigen::VectorXd phi_vec(nphi());
        for (int i = 0; i < nphi(); ++i) {
            phi_vec << phi[i];
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
            
            simplebounce::BounceCalculator bounce;
            bounce.setRmax(rmax);
            bounce.setDimension(4);
            bounce.setN(grid);

            // needs delete
            simplebounce::GenericModel *model = new SimpleBouncePotential(potential);

            // don't need delete
            const double *tv = true_vacuum.data();
            const double *fv = false_vacuum.data();

            bounce.setModel(model);
            bounce.setVacuum(tv, fv);
            bounce.solve();

            // Extract results
            int n_fields = potential.get_number_of_fields();

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

            delete model;
            return BouncePath(radii, fields, bounce.action());
        }

private:
    double rmax;
    int grid;
    
};



};

#endif