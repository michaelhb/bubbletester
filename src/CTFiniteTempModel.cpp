#include "CTFiniteTempModel.hpp"

#define UNUSED(expr) (void)(expr) // Just to suppress warnings

namespace BubbleTester {

double CTFiniteTempModel::V_tree(const Eigen::VectorXd &coords, double T = 0) {
    UNUSED(T);
    double x = coords(0);
    double y = coords(1);
    return 0.25*l1*pow(pow(x,2)-v2,2)+0.25*l2*pow(pow(y,2)-v2,2)-mu2*x*y;
}

double CTFiniteTempModel::V_daisy(const Eigen::VectorXd &coords, double T = 0) {
    UNUSED(coords);
    UNUSED(T);
    return 0.0;
}

std::size_t CTFiniteTempModel::get_Ndim() {
    return N_dim;
}

std::vector<double> CTFiniteTempModel::get_squared_boson_masses(
    const Eigen::VectorXd &coords, double T = 0) {
    UNUSED(T);
    double x = coords(0);
    double y = coords(1);
    double a=l1*(3.*x*x-v2);
    double b=l2*(3.*y*y-v2);
    double A=0.5*(a+b);
    double B=sqrt(0.25*pow((a-b),2)+mu2*mu2);
    double mb = y1*(x*x+y*y) + y2*x*y;
    std::vector<double> masses = {A + B, A - B, mb};
    return masses;
}

std::vector<int> CTFiniteTempModel::get_boson_dof() {
    return boson_dof;
}

std::vector<double> CTFiniteTempModel::get_boson_constants() {
    return boson_constants;
}

std::vector<double> CTFiniteTempModel::get_squared_fermion_masses(
    const Eigen::VectorXd &coords, double T = 0) {
    UNUSED(T);
    UNUSED(coords);
    std::vector<double> fm{};
    return fm;
}

std::vector<int> CTFiniteTempModel::get_fermion_dof() {
    return fermion_dof;
}

double CTFiniteTempModel::get_renormalization_scale() {
    return renorm_scale;
}

};

