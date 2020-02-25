#ifndef BUBBLETESTER_CASADIPOTENTIAL_HPP_INCLUDED
#define BUBBLETESTER_CASADIPOTENTIAL_HPP_INCLUDED

#include <casadi/casadi.hpp>
#include "GenericPotential.hpp"

namespace BubbleTester {

class CasadiPotential : public GenericPotential {
    
public:
    CasadiPotential(casadi::Function fV_, int n_fields_) : fV(fV_), n_fields(n_fields_) {
        using namespace casadi;
        MX phi = MX::sym("phi", n_fields);
        MX grad = gradient(MX::vertcat(fV(phi)), phi);
        MX hess = MX::hessian(MX::vertcat(fV(phi)), phi);
        
        fGrad = Function("fGrad", {phi}, {grad}, {"phi"}, {"gradV(phi)"});
        fHess = Function("fHess", {phi}, {hess}, {"phi"}, {"hessV(phi)"});

        GenericPotential::init();
    }

    virtual double operator()(const Eigen::VectorXd& coords) const override {
        using namespace casadi;
        Eigen::VectorXd coords_tr = transform_coords(coords);

        DM coords_in = std::vector<double>(
            coords_tr.data(), coords_tr.data() + coords_tr.cols()*coords_tr.rows());
        
        return transform_v(fV(coords_in)[0].get_elements()[0], true);
    }

    virtual double partial(const Eigen::VectorXd& coords, int i) const override {
        using namespace casadi;
        Eigen::VectorXd coords_tr = transform_coords(coords);

        if (!grad_cache_bad && grad_cache_l == coords_tr) {
            return transform_v((double) grad_cache_r(i));
        }
        else {
            grad_cache_bad = false;
            grad_cache_l = coords_tr;
            DM coords_dm = DM::vertcat(eigen_to_dmvec(coords_tr));
            DM grad = fGrad(coords_dm)[0];
            grad_cache_r = grad;

            return transform_v((double) grad_cache_r(i));
        }
    };


    virtual double partial(const Eigen::VectorXd& coords, int i, int j) const override {
        using namespace casadi;
        Eigen::VectorXd coords_tr = transform_coords(coords);

        if (!hess_cache_bad && hess_cache_l == coords_tr) {
            return transform_v((double) hess_cache_r(i, j));
        }
        else {
            hess_cache_bad = false;
            hess_cache_l = coords_tr;
            DM coords_dm = DM::vertcat(eigen_to_dmvec(coords_tr));
            DM hess = fHess(coords_dm)[0];
            hess_cache_r = hess; 
            return transform_v((double) hess_cache_r(i, j));
        }
    }
    
    virtual std::size_t get_number_of_fields() const override {
        return n_fields;
    }

    casadi::Function get_function() { return fV; }
private:
    casadi::Function fV;
    casadi::Function fGrad;
    casadi::Function fHess;

    int n_fields;

    // We'll cache one level of calls to the derivatives to 
    // prevent redundant finite difference calculations
    mutable bool grad_cache_bad = true; 
    mutable Eigen::VectorXd grad_cache_l{};
    mutable casadi::DM grad_cache_r{};
    mutable bool hess_cache_bad = true;
    mutable Eigen::VectorXd hess_cache_l{};
    mutable casadi::DM hess_cache_r{};

    casadi::DMVector eigen_to_dmvec(Eigen::VectorXd vec) const {
        casadi::DMVector dmVec;
        for (int i = 0; i < vec.size(); ++i) {
            dmVec.push_back(vec(i));
        }
        return dmVec;
    }
};

}

#endif