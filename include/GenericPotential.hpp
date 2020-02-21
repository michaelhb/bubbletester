#ifndef BUBBLETESTER_GENERICPOTENTIAL_HPP_INCLUDED
#define BUBBLETESTER_GENERICPOTENTIAL_HPP_INCLUDED

#include <Eigen/Core>
#include <exception>
#include <vector>
#include <math.h>
#include <nlopt.hpp>
#include "BouncePath.hpp"

// For 2D plots
typedef std::tuple<double, double, double> data_row; 
typedef std::tuple<double, double> point_marker;

namespace BubbleTester {

class GenericPotential {
public:
   
   //! Must be called in all subclass constructors
   void init() {
      std::size_t n_fields = get_number_of_fields();
      origin_translation_gp = Eigen::VectorXd::Zero(n_fields);
      basis_transform_gp = Eigen::MatrixXd::Identity(n_fields, n_fields);
   }
   
   //! Evaluate potential at point
   /*!
    * @param coords Coordinates at which to evaluate
    * @return Value of potential at coordinates
    */
   virtual double operator()(const Eigen::VectorXd& coords) const = 0;

   //! Partial derivative WRT coordinate i at a point
   /*!
    * @param coords Coordinates at which to evaluate
    * @param i Index of coordinate to be differentiated
    * @return Value of specified partial at point
    */
   virtual double partial(const Eigen::VectorXd& coords, int i) const = 0;

   //! Partial derivative WRT coordinates i, j at a a point
   /*!
    * @param coords Coordinates at which to evaluate
    * @param i Index of first coordinate to be differentiated
    * @param j Index of second coordinate to be differentiated
    * @return Value of specified partial at point
    */
   virtual double partial(const Eigen::VectorXd& coords, int i, int j) const = 0;

   //! Offset the potential by a constant. 
   void offset_potential(double v_offset_) {
      v_offset = v_offset_;
   }

   //! Scale the potential v -> v_scale*v. 
   void scale_potential(double v_scale_) {
      v_scale *= v_scale_;
   }

   //! Translate origin in field space. 
   void translate_origin(Eigen::VectorXd origin_translation_) {
      origin_translation_gp += origin_translation_;
   }

   //! Apply a change of basis to the field coordinates
   void apply_basis_change(const Eigen::MatrixXd& cob_matrix_) {
      // Note we use the inverse transform on incoming coordinates!
      basis_transform_gp = basis_transform_gp * (cob_matrix_.transpose());
   }

   virtual std::size_t get_number_of_fields() const = 0;

   //! Contour plot of the potential (2 field potentials only)
   void plot_2d(std::string title, unsigned int axis_size, double x_min, double x_max, 
      double y_min, double y_max,
      std::vector<point_marker> point_marks = std::vector<point_marker>(),
      BouncePath path = BouncePath());

   //! Contour plot with auto plot box around vacua (2 field potentials only)
   void plot_2d(std::string title, unsigned int axis_size, Eigen::VectorXd true_vac, 
      Eigen::VectorXd false_vac, double margin, BouncePath path = BouncePath());

   //! Normalise the potential so that v(phi_f) - v(phi_t) = 1,
   // V(phi_f) = 0, phi_f = 0, and |phi_f - phi_t| = 1.
   // Returns a multiplier which can be used to recover the 
   // original action. 
   static double normalise(GenericPotential& potential, int n_spatial_dimensions,
      Eigen::VectorXd true_vacuum, Eigen::VectorXd false_vacuum);

   //! Utility function for locating vacua
   Eigen::VectorXd minimise(const Eigen::VectorXd& start, 
      const Eigen::VectorXd& lower_bounds, const Eigen::VectorXd& upper_bounds);

   //! Objective function for minimizing with NLOpt. 
   double nlopt_objective(const std::vector<double> &x, std::vector<double> &grad, void* f_data);

protected:   
   //! Transform incoming coordinates.
   Eigen::VectorXd transform_coords(Eigen::VectorXd coords) const {
      return (basis_transform_gp*coords + origin_translation_gp);
   }

   //! Transform outgoing potential & derivative scalars. Use offset=false for derivs.
   double transform_v(double v, bool offset=false) const {
      if (offset) {
         return (v + v_offset)*v_scale;
      }
      else {
         return v*v_scale;
      }
   }

   // Scale constants for normalisation
   double v_scale = 1;
   double v_offset = 0;
   Eigen::VectorXd origin_translation_gp;
   Eigen::MatrixXd basis_transform_gp{};

   // Various utility methods for making the 2D plots
   std::vector<std::tuple<double, double, double>> get_2d_potential_grid(
      unsigned int axis_size, double x_min, double x_max, double y_min, double y_max);

   double find_minimum(std::vector<data_row> grid);
   void shift_to_zero(std::vector<data_row>& grid);


};

//! Wrapper function for NLOpt
double nlopt_wrapper(const std::vector<double> &x, std::vector<double> &grad, void* f_data);

};

#endif