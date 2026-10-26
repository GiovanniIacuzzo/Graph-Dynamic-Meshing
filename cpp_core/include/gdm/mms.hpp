#pragma once

#include "gdm/diagnostics.hpp"
#include "gdm/mesh_state.hpp"
#include "gdm/parameters.hpp"
#include "gdm/pde_model.hpp"

#include <cstddef>
#include <vector>

namespace gdm {

struct SpatialConvergenceRow {
    std::size_t n_subdivisions{0};
    double h{0.0};
    double rel_l2_error{0.0};
    double rel_h1_error{0.0};
    double order_l2{0.0};
    double order_h1{0.0};
};

struct TemporalConvergenceRow {
    double dt{0.0};
    double cauchy_l2_error{0.0};
    double order_l2{0.0};
};

// Moto analitico armonico dei nodi utilizzato nei test C1-C5 (tests/test_ale_gcl.py e test_ale_convergence.py)
std::vector<MeshState::Coord2D> analytical_mesh_motion(
    const std::vector<MeshState::Coord2D>& nodes_0,
    double t,
    double T = 0.5,
    double amplitude = 0.02);

std::vector<MeshState::Coord2D> analytical_mesh_velocity(
    const std::vector<MeshState::Coord2D>& nodes_0,
    double t,
    double T = 0.5,
    double amplitude = 0.02);

std::vector<SpatialConvergenceRow> run_fem_spatial_convergence_study(
    const std::vector<std::size_t>& grid_resolutions,
    const PDEParameters& pde_params = PDEParameters{},
    double t_final = 0.05,
    double base_dt_at_n8 = 1.0e-3);

std::vector<SpatialConvergenceRow> run_ale_spatial_convergence_study(
    const std::vector<std::size_t>& grid_resolutions,
    const PDEParameters& pde_params = PDEParameters{},
    double t_final = 0.05,
    double base_dt_at_n8 = 1.0e-3,
    double motion_amplitude = 0.02);

std::vector<TemporalConvergenceRow> run_ale_temporal_convergence_study(
    std::size_t grid_n,
    const std::vector<double>& dt_list,
    const PDEParameters& pde_params = PDEParameters{},
    double t_final = 0.1,
    double motion_amplitude = 0.02);

} // namespace gdm