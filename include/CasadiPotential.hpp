#ifndef BUBBLETESTER_CASADIPOTENTIAL_HPP_INCLUDED
#define BUBBLETESTER_CASADIPOTENTIAL_HPP_INCLUDED

#include <memory>
#include <casadi/casadi.hpp>
#include "GenericPotential.hpp"
#include "CasadiCommon.hpp"

namespace BubbleTester {

// Wrap an arbitrary GenericPotential in a casadi::Callback. 
// For now, we'll let CasADi handle the derivatives with 
// finite differences. 
class CasadiPotentialCallback : public casadi::Callback {
public:
    CasadiPotentialCallback(const GenericPotential& potential_) : 
        potential(potential_) {
            using namespace casadi;
            n_fields = potential.get_number_of_fields();
            Dict opts;
            opts["enable_fd"] = true;
            construct("PotentialCallback", opts);
        }

    casadi_int get_n_in() override { return 1; }
    casadi_int get_n_out() override { return 1; }

    casadi::Sparsity get_sparsity_in(casadi_int i) override {
        return casadi::Sparsity::dense(n_fields, 1);
    }

    std::vector<casadi::DM> eval(const std::vector<casadi::DM>& arg) const override {
        using namespace casadi;
        std::vector<double> vArg = arg.at(0).get_elements();
        Eigen::Map<Eigen::VectorXd> eArg(vArg.data(), vArg.size());
        return {potential(eArg)};
    }

private:
    const GenericPotential& potential;
    int n_fields;
};

// Wrap a potential specified as a casadi::Function in a GenericPotential.
// This can be used by all GenericBounceSolvers, but the CasadiSolver can
// access the underlying Function and use automatic differentiation.
class CasadiPotential : public GenericPotential {
   
public:
    CasadiPotential(casadi::Function fV_, int n_fields_, 
        casadi::SXVector params_ = {}, std::vector<double> params0_ = {}) : 
        fV(fV_), params(params_), n_fields(n_fields_), param_vals(params0_) {
        using namespace casadi;

        SX phi = SX::sym("phi", n_fields);
        SXDict argV;
        argV["phi"] = phi;
        for (int i = 0; i < params.size(); ++i) {
            argV[params[i].name()] = params[i];
        }

        SX grad = gradient(fV(argV).at("V"), phi);
        SX hess = SX::hessian(fV(argV).at("V"), phi);
        
        fGrad = Function("fGrad", {phi}, {grad}, {"phi"}, {"grad"});
        fHess = Function("fHess", {phi}, {hess}, {"phi"}, {"hess"});

        GenericPotential::init();
    }

    casadi::SXVector get_params() const {
        return params;
    }

    std::vector<double> get_param_vals() const {
        return param_vals;
    }

    void set_param_vals(std::vector<double> param_vals_) const {
        // I know, I know...
        param_vals = param_vals_;
        grad_cache_bad = true;
        hess_cache_bad = true;
    }

    void add_params(casadi::DMDict& args) const {
        for (int i = 0; i < params.size(); ++i) {
            args[params[i].name()] = param_vals[i];
        }
    } 

    virtual double operator()(const Eigen::VectorXd& coords) const override {
        using namespace casadi;
        Eigen::VectorXd coords_tr = transform_coords(coords);

        DMDict argV;
        argV["phi"] = eigen_to_dm(coords_tr);
        add_params(argV);

        DM res = fV(argV).at("V");
        return transform_v((double) res, true);
    }

    virtual double partial(const Eigen::VectorXd& coords, int i) const override {
        using namespace casadi;
        Eigen::VectorXd coords_tr = transform_coords(coords);
        if (!grad_cache_bad && grad_cache_l == coords_tr) {
            return transform_v((double) grad_cache_r(i), false);
        }
        else {
            DMDict argGrad;
            argGrad["phi"] = eigen_to_dm(coords_tr);
            add_params(argGrad);

            DM grad = fGrad(argGrad).at("grad");
            
            grad_cache_bad = false;
            grad_cache_r = grad;
            grad_cache_l = coords_tr;

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
            DMDict argHess;
            argHess["phi"] = eigen_to_dm(coords_tr);
            add_params(argHess);
            
            DM hess = fHess(argHess).at("hess");
            hess_cache_bad = false;
            hess_cache_r = hess; 
            hess_cache_l = coords_tr;
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
    casadi::SXVector params;
    mutable std::vector<double> param_vals;

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