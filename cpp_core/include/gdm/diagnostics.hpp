#pragma once

#include "gdm/mesh_metrics.hpp"
#include "gdm/mesh_state.hpp"
#include "gdm/pde_model.hpp"

#include <utility>
#include <vector>

namespace gdm {

struct ContinuousErrorNorms {
    double abs_l2_error{0.0};
    double rel_l2_error{0.0};
    double abs_h1_seminorm_error{0.0};
    double rel_h1_seminorm_error{0.0};
};

struct SimulationStepSummary {
    double time{0.0};
    ContinuousErrorNorms errors{};
    double discrete_mass{0.0};
    double min_jacobian{0.0};
    double mean_escobar_quality{0.0};
    std::size_t inverted_cells{0};
    double alpha_global{1.0};
};

ContinuousErrorNorms compute_continuous_errors(
    const MeshState& mesh,
    const std::vector<double>& u_h,
    const ManufacturedConvectionDiffusion2D& pde,
    double t);

double compute_mass_matrix_l2_difference(
    const MeshState& mesh,
    const std::vector<double>& u_a,
    const std::vector<double>& u_b);

double compute_time_integrated_error(
    const std::vector<double>& times,
    const std::vector<double>& errors);

double compute_free_stream_error_linf(const std::vector<double>& u_h,
                                      double reference_constant = 1.0);

double compute_relative_mass_drift(double initial_mass, double current_mass);

} // namespace gdm