#ifndef BUBBLETESTER_GENERICPOTENTIAL_HPP_INCLUDED
#define BUBBLETESTER_GENERICPOTENTIAL_HPP_INCLUDED

#include <Eigen/Core>
#include <exception>
#include <vector>

typedef std::tuple<double, double, double> data_row; // For 2D plots

namespace BubbleTester {

class GenericPotential {
public:
   
   //! Must be called in all subclass constructors
   void init() {
      origin_translation = Eigen::VectorXd::Zero(get_number_of_fields());
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

   //! Offset the potential by a constant. This is applied before scale_potential.
   void offset_potential(double v_offset_) {
      v_offset = v_offset_;
   }

   //! Scale the potential v -> v_scale*v. Applied after offset_potential.
   void scale_potential(double v_scale_) {
      v_scale *= v_scale_;
   }

   //! Scale coordinates phi -> phi_scale*phi. Applied after translate_origin.
   void scale_fields(double phi_scale_) {
      phi_scale *= phi_scale_;
   } 

   //! Translate origin in field space. Applied before scale_fields. 
   void translate_origin(Eigen::VectorXd origin_translation_) {
      origin_translation *= origin_translation_;
   }

   virtual std::size_t get_number_of_fields() const = 0;

   //! Contour plot of the potential (2 field potentials only)
   void plot_2d(std::string title, unsigned int axis_size, double x_min, double x_max, 
      double y_min, double y_max, double cutoff=-1.);

   //! Contour plot with auto plot box around vacua (2 field potentials only)
   void plot_2d(std::string title, unsigned int axis_size, Eigen::VectorXd true_vac, 
      Eigen::VectorXd false_vac, double margin, double cutoff=-1.);

protected:
   //! Transform incoming coordinates, first translating then scaling. 
   Eigen::VectorXd transform_coords(Eigen::VectorXd coords) const {
      return phi_scale * (coords + origin_translation);
   }

   //! Transform outgoing potential & derivative scalars. Use offset=false for derivs.
   double transform_v(double v, bool offset=false) const {
      if (offset) {
         return (v + offset)*v_scale;
      }
      else {
         return v*v_scale;
      }
   }

   // Scale constants for normalisation
   double v_scale = 1;
   double v_offset = 0;
   double phi_scale = 1;
   Eigen::VectorXd origin_translation;

   // Various utility methods for making the 2D plots
   std::vector<std::tuple<double, double, double>> get_2d_potential_grid(
      unsigned int axis_size, double x_min, double x_max, double y_min, double y_max);

   double find_minimum(std::vector<data_row> grid);
   void shift_to_zero(std::vector<data_row>& grid);
   void apply_cutoff(std::vector<data_row>& grid, double cutoff);

};

};

#endif