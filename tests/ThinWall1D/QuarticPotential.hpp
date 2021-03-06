#ifndef BUBBLETESTER_QUARTICPOTENTIAL_HPP_INCLUDED
#define BUBBLETESTER_QUARTICPOTENTIAL_HPP_INCLUDED

#include "GenericPotential.hpp"

namespace BubbleTester {

class QuarticPotential : public GenericPotential {
public:
    QuarticPotential(double alpha_) : alpha(alpha_) {
        if (alpha <= 0.5 || alpha >= 0.75) {
            throw "alpha must be between 0.5 and 0.75";
        }
        GenericPotential::init();
    }

    virtual double operator()(const Eigen::VectorXd& coords) const override;
    virtual double partial(const Eigen::VectorXd& coords, int i) const override;
    virtual double partial(const Eigen::VectorXd& coords, int i, int j) const override;
    
    virtual std::size_t get_number_of_fields() const override {
        return 1;
    }
private:
   double alpha{0.6};
   double E{1.};
};

double QuarticPotential::operator()(const Eigen::VectorXd& coords) const {
    double phip = transform_coords(coords)(0);
    
    double res = 0.5 * (3. - 4. * alpha) * E * phip * phip
        - E * phip * phip * phip
        + alpha * E * phip * phip * phip * phip;
    
    return transform_v(res, true);
}

double QuarticPotential::partial(const Eigen::VectorXd& coords, int i) const {
    double phip = transform_coords(coords)(0);

    double res = (3. - 4. * alpha) * E * phip - 3. * E * phip * phip
        + 4. * alpha * E * phip * phip * phip;

    return transform_v(res);
}

double QuarticPotential::partial(const Eigen::VectorXd& coords, int i, int j) const {
    double phip = transform_coords(coords)(0);

    double res = (3. - 4. * alpha) * E - 6. * E * phip
        + 12. * alpha * E * phip * phip;

    return transform_v(res);
}

};
#endif
