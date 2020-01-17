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
    double y_min, double y_max, double cutoff, std::vector<point_marker> point_marks) {

    if (get_number_of_fields() != 2) {
        throw std::invalid_argument("Can only get potential grid for 2 field potentials");
    }

    std::vector<data_row> grid = get_2d_potential_grid(axis_size, x_min, x_max, y_min, y_max);
    
    if (cutoff > 0) {
        apply_cutoff(grid, cutoff);
    }

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
}

void GenericPotential::plot_2d(std::string title, unsigned int axis_size,
     Eigen::VectorXd true_vac, Eigen::VectorXd false_vac, double margin, double cutoff) {
    double x_max = std::max(true_vac(0), false_vac(0)) + margin;
    double x_min = std::min(true_vac(0), false_vac(0)) - margin;
    double y_max = std::max(true_vac(1), false_vac(1)) + margin;
    double y_min = std::min(true_vac(1), false_vac(1)) - margin;

    std::vector<point_marker> vacua_marks;
    vacua_marks.push_back(std::make_tuple(true_vac(0), true_vac(1)));
    vacua_marks.push_back(std::make_tuple(false_vac(0), false_vac(1)));

    plot_2d(title, axis_size, x_min, x_max, y_min, y_max, cutoff, vacua_marks);

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

void GenericPotential::apply_cutoff(std::vector<data_row>& grid, double cutoff) {
    for (auto& row: grid) {
        row = std::make_tuple(
            std::get<0>(row),
            std::get<1>(row), 
            std::min(std::get<2>(row), cutoff));
    }
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
    Eigen::VectorXd shifted_true_vacuum = false_vacuum - true_vacuum;

    // Change the field basis so that the first component points to 
    // the true vacuum, and scale so that phi_t = (1,0,...,0)
    // double field_scaling = std::pow((false_vacuum - true_vacuum).norm(), -1);
    double field_scaling = (false_vacuum - true_vacuum).norm();
    Eigen::MatrixXd cob_matrix = calculate_rotation_to_target(shifted_true_vacuum);
    potential.apply_basis_change(field_scaling*cob_matrix);

    // Offset potential so that v(phi_f) = 0
    potential.offset_potential(-1.0*potential(origin));

    // // Scale potential so v(phi_f) - v(phi_t) = 1
    double potential_scaling = std::pow(v_phi_f - v_phi_t, -1);
    potential.scale_potential(potential_scaling);

    // Return the action rescaling factor
    return potential_scaling*std::pow(field_scaling, 2);
}

};