#ifndef BUBBLETESTER_EFFECTIVEPOTENTIAL_HPP_INCLUDED
#define BUBBLETESTER_EFFECTIVEPOTENTIAL_HPP_INCLUDED

#include "libeffpotential/effective_potential.hpp"
#include "GenericPotential.hpp"

namespace BubbleTester {

class FiniteTempPotential : public GenericPotential {
public:
    FiniteTempPotential(EffPotential::Abstract_input_model& model, double T_) 
        : potential(EffPotential::Effective_potential(model)), T(T_) {
        n_fields = potential.get_Ndim();
    }

    virtual double operator()(const Eigen::VectorXd& coords) const override;
    virtual double partial(const Eigen::VectorXd& coords, int i) const override;
    virtual double partial(const Eigen::VectorXd& coords, int i, int j) const override;
    
    virtual std::size_t get_number_of_fields() const override {
        return n_fields;
    }
private:
    double T;
    EffPotential::Effective_potential potential;
    std::size_t n_fields;

    // We'll cache one level of calls to the derivatives to 
    // prevent redundant finite difference calculations
    mutable bool grad_cache_bad = true; 
    mutable Eigen::VectorXd grad_cache_l{};
    mutable Eigen::VectorXd grad_cache_r{};
    mutable bool hess_cache_bad = true;
    mutable Eigen::VectorXd hess_cache_l{};
    mutable Eigen::MatrixXd hess_cache_r{};
};

};
#endif