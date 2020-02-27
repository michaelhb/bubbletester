#ifndef BUBBLETESTER_CASADIPOTENTIAL_HPP_INCLUDED
#define BUBBLETESTER_CASADIPOTENTIAL_HPP_INCLUDED

#include <memory>
#include <casadi/casadi.hpp>
#include "GenericPotential.hpp"

namespace BubbleTester {

casadi::DM eigen_to_dm(Eigen::VectorXd vec) {
    casadi::DMVector dmVec;
    for (int i = 0; i < vec.size(); ++i) {
        dmVec.push_back(vec(i));
    }
    return casadi::DM::vertcat(dmVec);
}

// Wrap an arbitrary GenericPotential in a casadi::Callback. 
// For now, we'll let CasADi handle the derivatives with 
// finite differences. 
class CasadiPotentialCallback : public casadi::Callback {
public:
    CasadiPotentialCallback(std::shared_ptr<const GenericPotential> potential_) : 
        potential(potential_) {
            n_fields = potential->get_number_of_fields();
            casadi::Dict opts;
            opts["enable_fd"] = true;
            construct("PotentialCallback", opts);
        }

    casadi_int get_n_in() override { return 1; }
    casadi_int get_n_out() override { return 1; }

    casadi::Sparsity get_sparsity_in(casadi_int i) {
        return casadi::Sparsity::dense(n_fields, 1);
    }

    std::vector<casadi::DM> eval(const std::vector<casadi::DM>& arg) const override {
        using namespace casadi;
        std::vector<double> vArg = arg.at(0).get_elements();
        Eigen::Map<Eigen::VectorXd> eArg(vArg.data(), vArg.size());
        return {potential->operator()(eArg)};
    }

private:
    std::shared_ptr<const GenericPotential> potential;
    int n_fields;
};

// Wrap a potential specified as a casadi::Function in a GenericPotential.
// This can be used by all GenericBounceSolvers, but the CasadiSolver can
// access the underlying Function and use automatic differentiation.
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
        DM coords_dm = eigen_to_dm(coords_tr);
        DM res = fV(coords_dm)[0];
        return transform_v((double) res, true);
    }

    virtual double partial(const Eigen::VectorXd& coords, int i) const override {
        using namespace casadi;
        Eigen::VectorXd coords_tr = transform_coords(coords);
        if (!grad_cache_bad && grad_cache_l == coords_tr) {
            return transform_v((double) grad_cache_r(i), false);
        }
        else {
            grad_cache_bad = false;
            grad_cache_l = coords_tr;
            DM coords_dm = eigen_to_dm(coords_tr);
            DM grad = fGrad(coords_dm)[0];
            grad_cache_r = grad;

            return transform_v((double) grad_cache_r(i), false);
        }
    };


    virtual double partial(const Eigen::VectorXd& coords, int i, int j) const override {
        using namespace casadi;
        Eigen::VectorXd coords_tr = transform_coords(coords);

        if (!hess_cache_bad && hess_cache_l == coords_tr) {
            return transform_v((double) hess_cache_r(i, j), false);
        }
        else {
            hess_cache_bad = false;
            hess_cache_l = coords_tr;
            DM coords_dm = eigen_to_dm(coords_tr);
            DM hess = fHess(coords_dm)[0];
            hess_cache_r = hess; 
            return transform_v((double) hess_cache_r(i, j), false);
        }
    }
    
    virtual std::size_t get_number_of_fields() const override {
        return n_fields;
    }

    casadi::Function get_function() const { return fV; }

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


};

}

#endif