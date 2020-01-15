#ifndef BUBBLETESTER_GENERICPOTENTIAL_HPP_INCLUDED
#define BUBBLETESTER_GENERICPOTENTIAL_HPP_INCLUDED

#include <Eigen/Core>
#include <exception>
#include <vector>

typedef std::tuple<double, double, double> data_row; // For 2D plots

namespace BubbleTester {

class GenericPotential {
public:
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

   virtual std::size_t get_number_of_fields() const = 0;

   //! Contour plot of the potential (2 field potentials only)
   void plot_2d(std::string title, unsigned int axis_size, double x_min, double x_max, 
      double y_min, double y_max, double cutoff=-1.);

   //! Contour plot with auto plot box around vacua (2 field potentials only)
   void plot_2d(std::string title, unsigned int axis_size, Eigen::VectorXd true_vac, 
      Eigen::VectorXd false_vac, double margin, double cutoff=-1.);


private:

   // Various utility methods for making the 2D plots

   std::vector<std::tuple<double, double, double>> get_2d_potential_grid(
      unsigned int axis_size, double x_min, double x_max, double y_min, double y_max);
   double find_minimum(std::vector<data_row> grid);
   void shift_to_zero(std::vector<data_row>& grid);
   void apply_cutoff(std::vector<data_row>& grid, double cutoff);

};

};

#endif