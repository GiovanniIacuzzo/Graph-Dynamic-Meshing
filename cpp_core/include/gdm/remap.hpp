#pragma once

#include "gdm/mesh_state.hpp"
#include "gdm/parameters.hpp"

#include <cstddef>
#include <vector>

namespace gdm {

struct RemapDiagnostics {
    std::vector<double> remapped_field;
    double mass_before{0.0};
    double mass_after{0.0};
    double relative_mass_drift{0.0};
    double elapsed_ms{0.0};
    std::size_t fallback_nearest_cell_count{0};
};

// Interpolazione baricentrica pointwise P1 della soluzione dalla mesh sorgente ai nodi della mesh destinazione
RemapDiagnostics remap_p1_solution(
    const MeshState& old_mesh,
    const std::vector<double>& old_u,
    const MeshState& new_mesh);

// Proiezione variazionale L2 globale: M(new_mesh) * u_new = \int_{\Omega} \phi_i^{new}(x) u_old(x) dx
RemapDiagnostics remap_l2_projection(
    const MeshState& old_mesh,
    const std::vector<double>& old_u,
    const MeshState& new_mesh,
    const LinearSolverParameters& solver_params = LinearSolverParameters{});

} // namespace gdm