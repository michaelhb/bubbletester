#include <algorithm>
#include <stdio.h>
#include <cmath>
#include "gnuplot-iostream.h"
#include "GenericPotential.hpp"
#include "Rotation.hpp"

namespace BubbleTester {

//! Utility method for plotting 2d potentials.
void GenericPotential::plot_2d(
    std::string title, unsigned int axis_size, double x_min, double x_max, 
    double y_min, double y_max, std::vector<point_marker> point_marks, BouncePath path) {

    if (get_number_of_fields() != 2) {
        throw std::invalid_argument("Can only get potential grid for 2 field potentials");
    }

    std::vector<data_row> grid = get_2d_potential_grid(axis_size, x_min, x_max, y_min, y_max);

    Gnuplot gp("tee /tmp/plot.gp | gnuplot -persist");
    gp << "reset\n";
    gp << "set dgrid3d 200,200,1\n";
    gp << "set contour base\n";
    gp << "unset surface\n";
    gp << "set cntrparam levels auto 100\n";
    gp << "set samples 250,2\n";
    gp << "set isosamples 2,250\n";
    gp << "set view map\n";
    gp << "set table '/tmp/contour.txt'\n";
    gp << "splot '-' using 1:2:(log10($3)) notitle\n";
    gp.send1d(grid);

    gp << "unset contour\n";
    gp << "unset table\n";
    gp << "unset dgrid3d\n";
    gp << "set autoscale fix\n";

    gp << "set title '" << title << "'\n";
    gp << "set xlabel 'x'\n";
    gp << "set ylabel 'y'\n";

    for (int i = 0; i < point_marks.size(); ++i) {
        gp << "set label " << i + 1 << " at " << std::get<0>(point_marks[i]) 
        << "," << std::get<1>(point_marks[i]) << " point pointtype 2 ps 5 front\n";
    }

    gp << "plot '-' u 1:2:3 w image not, '/tmp/contour.txt' u 1:2 w l not\n";
    gp << "plot '/tmp/contour.txt' u 1:2 w l not\n";
    gp.send1d(grid);

    // Eigen::MatrixXd profiles = path.get_profiles();

    // std::vector<point_marker> path_points;
    // for (int i = 0; i < profiles.rows(); ++i) {
    //     path_points.push_back(std::make_tuple(profiles(i,0), profiles(i,1)));
    // }

    // gp << "plot '-' with linespoints linestyle 1\n";
    // gp.send1d(path_points);
}

void GenericPotential::plot_2d(std::string title, unsigned int axis_size,
     Eigen::VectorXd true_vac, Eigen::VectorXd false_vac, double margin, BouncePath path) {
    double x_max = std::max(true_vac(0), false_vac(0)) + margin;
    double x_min = std::min(true_vac(0), false_vac(0)) - margin;
    double y_max = std::max(true_vac(1), false_vac(1)) + margin;
    double y_min = std::min(true_vac(1), false_vac(1)) - margin;

    std::vector<point_marker> vacua_marks;
    vacua_marks.push_back(std::make_tuple(true_vac(0), true_vac(1)));
    vacua_marks.push_back(std::make_tuple(false_vac(0), false_vac(1)));

    plot_2d(title, axis_size, x_min, x_max, y_min, y_max, vacua_marks, path);

}

void GenericPotential::shift_to_zero(std::vector<data_row>& grid) {
    double min = find_minimum(grid);

    for (auto& row: grid) {
        row = std::make_tuple(
            std::get<0>(row),
            std::get<1>(row),
            std::get<2>(row) - min);
    }
}

double GenericPotential::find_minimum(std::vector<data_row> grid) {
    double min = std::numeric_limits<double>::infinity();

    for (auto& row : grid) {
        double z = std::get<2>(row);
        if (z < min) min = z;
    }

    return min;
}

//! Get the data for 2d potential plots 
std::vector<std::tuple<double, double, double>> GenericPotential::get_2d_potential_grid(
    unsigned int axis_size, double x_min, double x_max, double y_min, double y_max) {
        
    if (get_number_of_fields() != 2) {
        throw std::invalid_argument("Can only get potential grid for 2 field potentials");
    }
    assert(x_min < x_max);
    assert(y_min < y_max);

    std::vector<std::tuple<double, double, double>> grid;

    double x_step = (x_max - x_min) / axis_size;
    double y_step = (y_max - y_min) / axis_size;

    for (unsigned int ix = 0; ix < axis_size; ix++) {
        for (unsigned int iy = 0; iy < axis_size; iy++) {
        double x = x_min + ix*x_step;
        double y = y_min + iy*y_step;
        Eigen::Vector2d eval_coord(x, y);
        double z = (*this)(eval_coord);
        grid.push_back(std::make_tuple(x, y, z));
        }
    }

    // Normalize so that the lowest potential value is zero
    shift_to_zero(grid);
    
    return grid;
}

double GenericPotential::normalise(GenericPotential& potential, 
      Eigen::VectorXd true_vacuum, Eigen::VectorXd false_vacuum) {
    
    Eigen::VectorXd origin = Eigen::VectorXd::Zero(potential.get_number_of_fields());
    double v_phi_t = potential(true_vacuum);
    double v_phi_f = potential(false_vacuum);

    double dist_true_vacuum = (false_vacuum - true_vacuum).norm();

    // Translate origin to true vacuum
    potential.translate_origin(false_vacuum);

    // Calc new position of true vacuum
    Eigen::VectorXd shifted_true_vacuum = true_vacuum - false_vacuum;

    // Change the field basis so that the first component points to 
    // the true vacuum, and scale so that phi_t = (1,0,...,0)
    double field_scaling = (false_vacuum - true_vacuum).norm();
    Eigen::MatrixXd cob_matrix = calculate_rotation_to_target(shifted_true_vacuum).transpose();
    potential.apply_basis_change(field_scaling*cob_matrix);

    // Offset potential so that v(phi_f) = 0
    potential.offset_potential(-1.0*potential(origin));

    // // Scale potential so v(phi_f) - v(phi_t) = 1
    double potential_scaling = std::pow(v_phi_f - v_phi_t, -1);
    potential.scale_potential(potential_scaling);

    // Return the action rescaling factor
    return 1.0 / (potential_scaling*std::pow(field_scaling, 2));
}

double nlopt_wrapper(const std::vector<double> &x, std::vector<double> &grad, void* f_data) {
   GenericPotential *potential = static_cast<GenericPotential *>(f_data);
   return potential->nlopt_objective(x, grad, f_data);
}

Eigen::VectorXd GenericPotential::minimise(
        const Eigen::VectorXd& start_, 
        const Eigen::VectorXd& lb_, const Eigen::VectorXd& ub_) {
            
    unsigned int n_fields = get_number_of_fields();
    
    std::vector<double> lb(lb_.data(), lb_.data() + lb_.size());
    std::vector<double> ub(ub_.data(), ub_.data() + ub_.size());
    
    nlopt::opt opt(nlopt::GN_DIRECT_L, n_fields);
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);

    opt.set_min_objective(nlopt_wrapper, this);
    opt.set_ftol_rel(1e-4);

    std::vector<double> res(start_.data(), start_.data() + start_.size());
    double res_val;

    opt.optimize(res, res_val);

    return Eigen::Map<Eigen::VectorXd>(res.data(), res.size());
}

double GenericPotential::nlopt_objective(const std::vector<double> &x, std::vector<double> &grad, void* f_data) {
    unsigned int n_fields = get_number_of_fields();
    
    Eigen::VectorXd coords = Eigen::Map<const Eigen::VectorXd>(x.data(), x.size());

    if (!grad.empty()) {
        for (int j = 0; j < n_fields; ++j) {
            grad[j] = partial(coords, j);
        }
    }

    return this->operator()(coords);
}

};