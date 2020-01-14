#ifndef BUBBLETESTER_BP1DRIVER_HPP_INCLUDED
#define BUBBLETESTER_BP1DRIVER_HPP_INCLUDED

#include "GenericBounceSolver.hpp"
#include "GenericPotential.hpp"
#include "BouncePath.hpp"

#include "BubbleProfiler/profile_guesser.hpp"
#include "BubbleProfiler/kink_profile_guesser.hpp"
#include "BubbleProfiler/field_profiles.hpp"
#include "BubbleProfiler/perturbative_profiler.hpp"
#include "BubbleProfiler/potential.hpp"
#include "BubbleProfiler/observers.hpp"

namespace BubbleTester {

//////// Wrapper class for BubbleProfiler::Potential 

class BubbleProfilerPotential : public BubbleProfiler::Potential {
public:
    virtual ~BubbleProfilerPotential() = default;
    BubbleProfilerPotential() = default;

    BubbleProfilerPotential(const GenericPotential& potential_) :
        potential(potential_) {
        n_fields = potential.get_number_of_fields();
        origin = Eigen::VectorXd::Zero(n_fields);
        origin_translation = Eigen::VectorXd::Zero(n_fields);
        basis_transform = Eigen::MatrixXd::Identity(n_fields, n_fields);
    }

    virtual BubbleProfilerPotential * clone() const override {
        return new BubbleProfilerPotential(*this);
    };

    virtual double operator()(const Eigen::VectorXd& coords) const override;
    virtual double partial(const Eigen::VectorXd& coords, int i) const override;
    virtual double partial(const Eigen::VectorXd& coords, int i, int j) const override;
    virtual std::size_t get_number_of_fields() const override;

    virtual void translate_origin(const Eigen::VectorXd&) override;
    virtual void apply_basis_change(const Eigen::MatrixXd&) override;
    virtual void add_constant_term(double) override;



private:
    const GenericPotential& potential;
    std::size_t n_fields;

    Eigen::VectorXd origin{};
    Eigen::VectorXd origin_translation{};
    Eigen::MatrixXd basis_transform{};
    double constant_term = 0;
};

double BubbleProfilerPotential::operator()(const Eigen::VectorXd& coords) const {
    Eigen::VectorXd internal_coords =
      (basis_transform * coords) + origin_translation;
    
    return potential(internal_coords);
}

double BubbleProfilerPotential::partial(const Eigen::VectorXd& coords, int i) const {
    Eigen::VectorXd internal_coords =
      (basis_transform * coords) + origin_translation;

    return potential.partial(internal_coords, i);
}

double BubbleProfilerPotential::partial(const Eigen::VectorXd& coords, int i, int j) const {
    Eigen::VectorXd internal_coords =
      (basis_transform * coords) + origin_translation;

    return potential.partial(internal_coords, i, j);
}

std::size_t BubbleProfilerPotential::get_number_of_fields() const {
    return potential.get_number_of_fields();
}

void BubbleProfilerPotential::translate_origin(const Eigen::VectorXd& translation) {
    origin_translation = translation;
}

void BubbleProfilerPotential::apply_basis_change(const Eigen::MatrixXd& new_basis) {
    basis_transform = basis_transform * (new_basis.transpose());
}

void BubbleProfilerPotential::add_constant_term(double constant) {
    constant_term += constant;
}



//////// Wrapper class for the BP1 perturbative profiler 

class BP1BounceSolver : public GenericBounceSolver {
public:
    BouncePath solve(
        const Eigen::VectorXd& true_vacuum,
        const Eigen::VectorXd& false_vacuum,
        const GenericPotential& potential) const override {
            using namespace BubbleProfiler;

            BubbleProfilerPotential bp_potential(potential);

            std::size_t n_fields = bp_potential.get_number_of_fields();

            std::shared_ptr<Kink_profile_guesser> kink_guesser
                = std::make_shared<Kink_profile_guesser>();
            std::shared_ptr<Profile_guesser> guesser(kink_guesser);

            // Need to shift origin to false vacuum...
            bp_potential.translate_origin(false_vacuum); 
            Eigen::VectorXd origin = Eigen::VectorXd::Zero(n_fields);
            Eigen::VectorXd shifted_true_vacuum = true_vacuum - false_vacuum;

            double domain_start = -1;
            double domain_end = -1;
            double initial_step_size = 0.1;
            double interpolation_fraction = 0.1;

            Field_profiles ansatz = guesser->get_profile_guess(
                bp_potential, shifted_true_vacuum, n_fields, domain_start, domain_end,
                initial_step_size, interpolation_fraction);

            RK4_perturbative_profiler profiler;
            profiler.set_domain_start(ansatz.get_domain_start());
            profiler.set_domain_end(ansatz.get_domain_end());
            profiler.set_initial_step_size(initial_step_size);
            profiler.set_interpolation_points_fraction(interpolation_fraction);
            profiler.set_true_vacuum_loc(shifted_true_vacuum);
            profiler.set_false_vacuum_loc(origin);
            profiler.set_initial_guesser(guesser);

            auto convergence_tester = std::make_shared<Relative_convergence_tester>();

            convergence_tester->set_max_iterations(30);
            profiler.set_convergence_tester(convergence_tester);

            Dummy_observer observer;
            profiler.calculate_bubble_profile(bp_potential, observer);

            Field_profiles profiles = profiler.get_bubble_profile();

            return BouncePath(
                profiles.get_spatial_grid(),
                profiles.get_field_profiles(),
                profiler.get_euclidean_action());
        }

        void set_verbose(bool verbose) override {
            using namespace BubbleProfiler;
            auto& logging_manager = logging::Logging_manager::get_manager();
            if (verbose) {
                logging_manager.set_minimum_log_level(logging::Log_level::Trace);
            } else {
                logging_manager.set_minimum_log_level(logging::Log_level::Warning);
            }
        }
};

};

#endif