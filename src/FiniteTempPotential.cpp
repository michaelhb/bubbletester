#include "FiniteTempPotential.hpp"
#include <Eigen/Core>

namespace BubbleTester {

double FiniteTempPotential::operator()(const Eigen::VectorXd& coords) const {
    Eigen::VectorXd internal_coords = transform_coords(coords);
    double res = const_cast<EffPotential::Effective_potential&>(potential).V(internal_coords, T);
    return transform_v(res, true);
}

double FiniteTempPotential::partial(const Eigen::VectorXd& coords, int i) const {
    Eigen::VectorXd internal_coords = transform_coords(coords);

    if (!grad_cache_bad && grad_cache_l == internal_coords) {
        return transform_v(grad_cache_r(i));
    }
    else {
        grad_cache_bad = false;
        grad_cache_l = internal_coords;
        Eigen::VectorXd grad = 
            const_cast<EffPotential::Effective_potential&>(potential).d2V_dxdt(internal_coords, T);
        grad_cache_r = grad;

        return transform_v(grad_cache_r(i));
    }
}

double FiniteTempPotential::partial(const Eigen::VectorXd& coords, int i, int j) const {
    Eigen::VectorXd internal_coords = transform_coords(coords);

    if (!hess_cache_bad && hess_cache_l == internal_coords) {
        return transform_v(hess_cache_r(i, j));
    } 
    else {
        hess_cache_bad = false;
        hess_cache_l = internal_coords;
        Eigen::MatrixXd hess = 
            const_cast<EffPotential::Effective_potential&>(potential).d2V_dx2(internal_coords, T);
        hess_cache_r = hess;

        return transform_v(hess_cache_r(i, j));
    }
}

};