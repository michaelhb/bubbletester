#ifndef BUBBLETESTER_CTANALYTICPOTENTIAL_HPP_INCLUDED
#define BUBBLETESTER_CTANALYTICPOTENTIAL_HPP_INCLUDED

#include <cmath>
#include <exception>
#include "GenericPotential.hpp"

#define sq(x) std::pow(x, 2)

namespace BubbleTester {

class CTAnalyticPotential : public GenericPotential {
public:
    CTAnalyticPotential(double delta_) {
        delta = delta_;
        GenericPotential::init();
    }
    virtual double operator()(const Eigen::VectorXd& coords) const override;
    virtual double partial(const Eigen::VectorXd& coords, int i) const override;
    virtual double partial(const Eigen::VectorXd& coords, int i, int j) const override;
    
    virtual std::size_t get_number_of_fields() const override {
        return 2;
    }
private:
    double delta;
};

double CTAnalyticPotential::operator()(const Eigen::VectorXd& coords) const {
    
    Eigen::VectorXd internal_coords = transform_coords(coords);
    double x = internal_coords(0);
    double y = internal_coords(1);

    double res = (sq(x) + sq(y))*(1.8*sq(x - 1) + 0.2*sq(y - 1) - delta);
    
    return transform_v(res, true);
}

double CTAnalyticPotential::partial(const Eigen::VectorXd& coords, int i) const {
    
    Eigen::VectorXd internal_coords = transform_coords(coords);
    double x = internal_coords(0);
    double y = internal_coords(1);

    double res;

    if (i == 0) {
        res = 3.6*(x - 1)*(sq(x) + sq(y)) + 2*x*(1.8*sq(x - 1) + 0.2*sq(y - 1) - delta);
    }
    else if (i == 1) {
        res = 0.4*(y - 1)*(sq(x) + sq(y)) + 2*y*(1.8*sq(x - 1) + 0.2*sq(y - 1) - delta);
    }
    else {
        throw std::invalid_argument("This is a 2 field potential!");
    }

    return transform_v(res, false);
}

double CTAnalyticPotential::partial(const Eigen::VectorXd& coords, int i, int j) const {
    
    Eigen::VectorXd internal_coords = transform_coords(coords);
    double x = internal_coords(0);
    double y = internal_coords(1);

    double res;

    if (i == 0 && j == 0) {
        res = 3.6*(sq(x) + sq(y)) + 2*(1.8*sq(x - 1) + 0.2*sq(y - 1) - delta) + 14.4*(x - 1)*x;
    }
    else if (i == 1 && j == 1) {
        res = 0.4*(sq(x) + sq(y)) + 2*(1.8*sq(x - 1) + 0.2*sq(y - 1) - delta) + 1.6*(y - 1)*y;
    }
    else if ((i == 0 && j == 1) || (i == 1 && j == 0)) {
        res = 0.8*x*(y - 1) + 7.2*(x - 1)*y;
    }
    else {
        throw std::invalid_argument("This is a 2 field potential!");
    }

    return transform_v(res, false);
}

};

#endif
