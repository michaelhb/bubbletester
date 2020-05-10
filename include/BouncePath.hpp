#ifndef BUBBLETESTER_BOUNCEPATH_HPP_INCLUDED
#define BUBBLETESTER_BOUNCEPATH_HPP_INCLUDED

#include <Eigen/Core>
#include "gnuplot-iostream.h"

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

    void plot_profiles(double r_max = -1., std::string title = "") const {
        int n_phi = profiles.cols();
        int n_rho = profiles.rows();

        std::vector<std::vector<double>> profile_data;

        for (int i = 0; i < n_rho; ++i) {
            if (r_max > 0 && radii(i) > r_max) {
                break;
            }
            std::vector<double> row = {radii(i)};
            for (int j = 0; j < n_phi; ++j) {
                row.push_back(profiles(i, j));
            }
            profile_data.push_back(row);
        }        

        Gnuplot gp;
        gp << "set title '" << title << "'\n";
        gp << "plot '-' using 1:2 with lines title 'phi_1'";
        for (int i = 2; i <= n_phi; ++i) {
            gp << ", '' using 1:" << i + 1 << " with lines title 'phi_" << i << "'";
        }
        gp << "\n";

        for (int i = 0; i < n_phi; ++i) {
            gp.send1d(profile_data);
        }        
    };

private:
    double action;
    Eigen::VectorXd radii;
    Eigen::MatrixXd profiles;
};

};

#endif