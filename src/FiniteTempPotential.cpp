#include "FiniteTempPotential.hpp"
#include <Eigen/Core>

namespace BubbleTester {

double FiniteTempPotential::operator()(const Eigen::VectorXd& coords) const {
    return const_cast<EffPotential::Effective_potential&>(potential).V(coords, T);
}

double FiniteTempPotential::partial(const Eigen::VectorXd& coords, int i) const {

    if (!grad_cache_bad && grad_cache_l == coords) {
        return grad_cache_r(i);
    }
    else {
        grad_cache_bad = false;
        grad_cache_l = coords;
        Eigen::VectorXd grad = 
            const_cast<EffPotential::Effective_potential&>(potential).d2V_dxdt(coords, T);
        grad_cache_r = grad;

        return grad_cache_r(i);
    }
}

double FiniteTempPotential::partial(const Eigen::VectorXd& coords, int i, int j) const {

    if (!hess_cache_bad && hess_cache_l == coords) {
        return hess_cache_r(i, j);
    } 
    else {
        hess_cache_bad = false;
        hess_cache_l = coords;
        Eigen::MatrixXd hess = 
            const_cast<EffPotential::Effective_potential&>(potential).d2V_dx2(coords, T);
        hess_cache_r = hess;

        return hess_cache_r(i, j);
    }
}

};