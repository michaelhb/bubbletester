#ifndef BUBBLETESTER_BOUNCEPATH_HPP_INCLUDED
#define BUBBLETESTER_BOUNCEPATH_HPP_INCLUDED

#include <Eigen/Core>

namespace BubbleTester {

class BouncePath {
public:
    BouncePath() = default;
    
    BouncePath(const Eigen::VectorXd& radii_,
               const Eigen::MatrixXd& profiles_,
               double action_) {
        assert(radii_.rows() == profiles_.rows());
        radii = radii_;
        profiles = profiles_;
        action = action_;
    };

    // Just expose the data for now, add more appropriate 
    // accessors later as required.

    const Eigen::VectorXd& get_radii() const {return radii;}

    const Eigen::MatrixXd& get_profiles() const {return profiles;}

    double get_action() const {return action;}

private:
    double action;
    Eigen::VectorXd radii;
    Eigen::MatrixXd profiles;
};

};

#endif